# FreeCompose/run_demo.py（主执行文件，整合第一段代码）
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"  # 第一段环境变量配置
from typing import Tuple, Union, Optional, List
import shutil
import copy
import math
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import _preprocess_adapter_image
from transformers import CLIPProcessor, CLIPModel
from accelerate.utils import set_seed
import numpy as np
from PIL import Image
import cv2
from tqdm.notebook import tqdm
from IPython.display import display, clear_output
from sklearn.decomposition import PCA

# 设备配置（原第一段末尾）
device = torch.device('cuda:0')
dtype = torch.float16

# 导入现有文件中的模块（不修改原文件）
from utils.kvctrl import KVReplace, KVSelfReplace
from utils.masactrl import MutualSelfAttentionControlMaskAuto
from utils.attention import register_attention_editor_diffusers, AttentionBase

model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "pretrained/stable-diffusion-v2-1"
# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "pretrained/AnyLoRA"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
pipeline.enable_xformers_memory_efficient_attention()


def image_optimization(pipe: StableDiffusionPipeline, image: np.ndarray, text_source: str, text_target: str,
                       num_iters=200, use_dds=True) -> None:
    dds_loss = DDSLoss(device, pipe)
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text = get_text_embeddings(pipeline, text_source)
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    image_target.requires_grad = True

    z_target = z_source.clone()
    z_target.requires_grad = True
    optimizer = SGD(params=[z_target], lr=1e-1)

    for i in range(num_iters):
        if use_dds:
            loss, log_loss = dds_loss.get_dds_loss(z_source, z_target, embedding_source, embedding_target)
        else:
            loss, log_loss = dds_loss.get_sds_loss(z_target, embedding_target)
        optimizer.zero_grad()
        loss *= 10
        (2000 * loss).backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            out = decode(z_target, pipeline, im_cat=image)
            clear_output(wait=True)
            display(out)

    return decode(z_target, pipeline)


def edit_image_optimization(pipe: StableDiffusionPipeline, image: np.ndarray, text_source: str, text_target: str,
                            kind="sketch", num_iters=200, STEP=0, LAYER=10, cond_images=None, cond_scale=0.9,
                            append_images=None) -> None:
    pipe2 = copy.deepcopy(pipe)
    editor = KVReplace(STEP, LAYER, total_steps=num_iters)
    register_attention_editor_diffusers(pipe2, editor)

    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2iadapter_sketch_sd15v2" if kind == "sketch" else "TencentARC/t2iadapter_canny_sd15v2",
        torch_dtype=dtype,
    ).to(device)
    dds_loss = DDSLoss(device, pipe2, adapter=adapter, cond_images=cond_images, cond_scale=cond_scale)
    if kind == "sketch":
        dds_loss.t_min = 100
        dds_loss.t_max = 600

    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text = get_text_embeddings(pipeline, text_source)
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    image_target.requires_grad = True

    z_target = z_source.clone()
    z_target.requires_grad = True
    z_target_original = z_target.clone().detach()
    z_target_original.requires_grad = False
    optimizer = SGD(params=[z_target], lr=5e-2)

    for i in range(num_iters):
        if i == num_iters - 50:
            dds_loss.t_min = 50
            dds_loss.t_max = 150

        origin_loss, log_loss = dds_loss.get_dds_loss(z_source, z_taregt, embedding_source, embedding_target)

        loss = 10 * origin_loss

        optimizer.zero_grad()
        (2000 * loss).backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            out = decode(z_target, pipeline, im_cat=[image] + append_images if append_images is not None else [image])
            clear_output(wait=True)
            display(out)

    del pipe2

    return decode(z_target, pipeline)

case="01"
image = load_512(f"assets/cases/composition/{case}/source.jpg")
cond1 = load_512_mask(f"assets/cases/composition/{case}/origin_sketch.jpg") * 255
cond2 = load_512_mask(f"assets/cases/composition/{case}/edit_sketch.jpg") * 255
cond_images = [Image.fromarray(cond1), Image.fromarray(cond2)]

edit_image_optimization(
    pipeline, image,
    "A toy.", "A toy.",
    num_iters=300,
    STEP=0, LAYER=12,
    cond_images=cond_images,
    cond_scale=1.5,
    append_images=[cond1, cond2]
)

