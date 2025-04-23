import torch
from torch import nn
from diffusers import UNet2DConditionModel
from typing import Tuple


def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        # print(timestep)
        if timestep is not None and isinstance(timestep, int):
            timestep = torch.tensor([timestep] * z.shape[0], device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, get_raw=False,
                           guidance_scale=7.5):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        # [2, 2, 77, 768] -> [4, 77, 768]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                latent_input,
                timestep,
                embedd,
                down_intrablock_additional_residuals=[state.clone() for state in
                                                      self.adapter_state] if self.use_adapter else None,
            ).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def get_sds_loss(self, z: T, text_embeddings: T, eps: TN = None, mask=None, t=None,
                     timestep: Optional[int] = None, guidance_scale=100) -> TS:
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, _ = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        del grad_z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss

    def get_dds_loss(self, z_source: T, z_target: T, text_emb_source: T, text_emb_target: T,
                     eps=None, reduction='mean', symmetric: bool = False, calibration_grad=None,
                     timestep: Optional[int] = None,
                     guidance_scale=7.5, raw_log=False) -> TS:
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(z_source, eps, timestep)
            z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
            eps_pred, _ = self.get_eps_prediction(torch.cat((z_t_source, z_t_target)),
                                                  torch.cat((timestep, timestep)),
                                                  torch.cat((text_emb_source, text_emb_target)),
                                                  torch.cat((alpha_t, alpha_t)),
                                                  torch.cat((sigma_t, sigma_t)),
                                                  guidance_scale=guidance_scale)
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source)
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timestep - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()
        loss = z_target * grad.clone()
        if symmetric:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss

    def get_masked_dds_loss(self, z_source: T, z_target: T, text_emb_source: T, text_emb_target: T,
                            eps=None, reduction='mean', symmetric: bool = False, calibration_grad=None,
                            timestep: Optional[int] = None,
                            guidance_scale=7.5, raw_log=False, mask=None) -> TS:
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(z_source, eps, timestep)
            z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
            eps_pred, _ = self.get_eps_prediction(torch.cat((z_t_source, z_t_target)),
                                                  torch.cat((timestep, timestep)),
                                                  torch.cat((text_emb_source, text_emb_target)),
                                                  torch.cat((alpha_t, alpha_t)),
                                                  torch.cat((sigma_t, sigma_t)),
                                                  guidance_scale=guidance_scale)
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source)
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timestep - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()
        loss = z_target * grad.clone()
        if symmetric:
            if mask == None:
                loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            else:
                loss = (loss * mask).sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            if mask == None:
                loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            else:
                loss = (loss * mask).sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss

    def get_masked_inpaint_dds_loss(self, z_source: T, z_target: T, text_emb_source: T, text_emb_target: T,
                                    eps=None, reduction='mean', symmetric: bool = False, calibration_grad=None,
                                    timestep: Optional[int] = None,
                                    guidance_scale=7.5, raw_log=False, mask=None, iteration=3) -> TS:
        mean_pool = nn.AvgPool2d(5, stride=1, padding=2)

        def mean_n(origin, iteration):
            for _ in range(iteration):
                origin[mask < 1] = (mean_pool(origin) * (1 - mask))[mask < 1]
            return origin

        z_source = mean_n(z_source, iteration)

        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(z_source, eps, timestep)
            z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
            eps_pred, _ = self.get_eps_prediction(torch.cat((z_t_source, z_t_target)),
                                                  torch.cat((timestep, timestep)),
                                                  torch.cat((text_emb_source, text_emb_target)),
                                                  torch.cat((alpha_t, alpha_t)),
                                                  torch.cat((sigma_t, sigma_t)),
                                                  guidance_scale=guidance_scale)
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source)
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timestep - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()
        loss = z_target * grad.clone()
        if symmetric:
            if mask == None:
                loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            else:
                loss = (loss * mask).sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            if mask == None:
                loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            else:
                loss = (loss * mask).sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss

    def change_cond_scale(self, scale: float):
        self.adapter_state = [state * scale for state in self.origin_adapter_state]

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=dtype, adapter: T2IAdapter = None,
                 cond_images: List[Image.Image] = None, cond_scale: float = 0.9):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.prediction_type

        self.use_adapter = False
        if adapter is not None and cond_images is not None:
            self.use_adapter = True

            assert len(cond_images) == 2

            image_inputs = [_preprocess_adapter_image(image, 512, 512).to(device=device, dtype=dtype) for image in
                            cond_images]
            adapter_states = [adapter(image_input) for image_input in image_inputs]
            self.origin_adapter_state = adapter_states[0]
            self.adapter_state = adapter_states[0]
            for k in range(len(adapter_states[0])):
                v0 = adapter_states[0][k]
                v1 = adapter_states[1][k]
                self.origin_adapter_state[k] = torch.cat([v0, v1, v0, v1], dim=0)
                self.adapter_state[k] = self.origin_adapter_state[k] * cond_scale