from typing import Tuple, Union, Optional, List
# FreeCompose/utils/new_decode.py（新建，包含第二段代码）
from typing import Tuple, Union, Optional, List
import numpy as np
import torch
from PIL import Image


def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def load_512_mask(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))
    if len(image.shape) == 3:
        image = image[:, :, :3]
    else:
        image = np.stack([image, image, image], axis=2)
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)).convert('L'))
    # to binary
    image = (image > 100).astype(np.uint8)
    return image


@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                            return_tensors="pt", return_overflowing_tokens=True).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline, im_cat: TN = None):
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        if not isinstance(im_cat, list):
            im_cat = [im_cat]
        # change 2 dim to 3 dim
        im_cat = [np.repeat(np.expand_dims(im, 2), 3, axis=2) if len(im.shape) == 2 else im for im in im_cat]
        image = np.concatenate(im_cat + [image], axis=1)
    return Image.fromarray(image)


def decode_latent(latent: T, pipe: StableDiffusionPipeline):
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # image = denormalize(image)
    return image


def visualize_pca(feature_maps_fit_data: torch.Tensor, feature_maps_transform_data: torch.Tensor):
    feature_maps_fit_data = feature_maps_fit_data.reshape(1, -1).cpu().numpy()
    print(feature_maps_fit_data.shape)
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.reshape(1, -1).cpu().numpy())  # N X 3
    # print(feature_maps_pca)
    # feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), num_frame, -1, 1).repeat(3, axis=-1) # K x F x (H * W) x 3
    feature_maps_pca = feature_maps_pca.reshape(1, -1, 3)  # 1 x (H * W) x 3
    # print(feature_maps_pca.shape)
    H, W = math.sqrt(feature_maps_pca.shape[1]), math.sqrt(feature_maps_pca.shape[1])
    H, W = int(H), int(W)
    feature_maps_pca = feature_maps_pca.reshape(H, W, 3)

    return feature_maps_pca