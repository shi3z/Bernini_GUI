# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end Bernini Renderer inference pipeline: preprocess -> sample -> decode -> save."""

import html
import logging
import os
import re

import ftfy
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer

from .data_utils import make_divisible, preprocess_image, preprocess_video
from .io_utils import save_output
from .models import BerniniRendererConfig, BerniniRendererModel
from .weights import load_weights

logger = logging.getLogger("bernini.pipeline")


def _prompt_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _vae_encode(vae, x: torch.Tensor) -> torch.Tensor:
    """Encode `[1,C,T,H,W]` pixels into normalized VAE latents."""
    latents = vae.encode(x).latent_dist.mode()
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(1, z, 1, 1, 1)
    return (latents - mean) / std


def _vae_decode(vae, latents: torch.Tensor):
    """Decode VAE latents into a numpy clip `[T, H, W, C]` in [0, 1]."""
    latents = latents.to(vae.dtype)
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
    latents = latents * std + mean
    video = vae.decode(latents, return_dict=False)[0]
    processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    return processor.postprocess_video(video, output_type="np")[0]


class BerniniRendererPipeline:
    """Loads the model once; each call generates one video / image."""

    def __init__(self, model, vae, tokenizer, device):
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config_dir: str,
        high_noise_ckpt: str = None,
        low_noise_ckpt: str = None,
        device="cuda",
        load_ckpt_weights: bool = True,
        **config_overrides,
    ) -> "BerniniRendererPipeline":
        config = BerniniRendererConfig.from_pretrained(config_dir, **config_overrides)
        tokenizer = AutoTokenizer.from_pretrained(
            config.wan22_base, subfolder="tokenizer", trust_remote_code=True
        )
        vae = AutoencoderKLWan.from_pretrained(config.wan22_base, subfolder="vae", torch_dtype=torch.float32)
        vae.eval()
        vae.requires_grad_(False)
        model = BerniniRendererModel(config)
        if load_ckpt_weights:
            load_weights(model, high_noise_ckpt, low_noise_ckpt)
        model.eval()
        return cls(model, vae, tokenizer, device)

    def _tokenize(self, prompt: str):
        out = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return out.input_ids, out.attention_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        neg_prompt: str = "",
        num_frames: int = 81,
        max_image_size: int = 624,
        height: int = 480,
        width: int = 832,
        video=None,
        image=None,
        images=None,
        num_inference_steps: int = 40,
        guidance_mode: str = "rv2v",
        omega_V: float = 3.0,
        omega_I: float = 3.0,
        omega_TI: float = 4.0,
        omega_scale: float = 0.75,
        flow_shift: float = 5.0,
        seed: int = 42,
        fps: int = 16,
        eta: float = 0.5,
        norm_threshold=(50.0, 50.0),
        momentum: float = -0.5,
        system_prompt: str = "",
        output_path: str = "output.mp4",
        write_output: bool = True,
        progress_callback=None,
    ):
        """Generate one clip and write it to `output_path`.

        `video` drives video editing, `image` a single-image edit, `images` a
        list of reference images; the output size follows the source video or
        single image, otherwise `height` / `width`.

        With `write_output=False` the decode/save step is skipped (used by the
        redundant ranks of an Ulysses group) and ``None`` is returned.
        """
        device = self.device
        prompt = system_prompt + _prompt_clean(prompt)
        logger.info("prompt: %s", prompt)
        prompt_ids, prompt_mask = self._tokenize(prompt)
        neg_ids, neg_mask = self._tokenize(neg_prompt)

        # ---- encode visual conditions on the VAE ----
        self.vae.to(device)
        t, h, w = num_frames, None, None

        multi_video_vae_latents = None
        if video is not None:
            paths = video if isinstance(video, list) else [video]
            multi_video_vae_latents = []
            first_shape = None
            for vp in paths:
                pv = preprocess_video(
                    vp, fps=fps, max_image_size=max_image_size, max_image_num=num_frames, device=device
                )
                if first_shape is None:
                    first_shape = pv.shape
                multi_video_vae_latents.append(_vae_encode(self.vae, pv))
            t, h, w = first_shape[-3], first_shape[-2], first_shape[-1]

        image_vae_latents = None
        if image is not None:
            pi = preprocess_image(image, max_image_size=max_image_size, device=device)
            if h is None:
                h, w = pi.shape[-2], pi.shape[-1]
            image_vae_latents = _vae_encode(self.vae, pi)

        multi_image_vae_latents = None
        if images:
            multi_image_vae_latents = [
                _vae_encode(self.vae, preprocess_image(img, max_image_size=max_image_size, device=device))
                for img in images
            ]

        self.vae.to("cpu")
        torch.cuda.empty_cache()

        if h is None:
            h, w = height, width
        h, w = make_divisible(h, 16), make_divisible(w, 16)

        # ---- diffusion sampling ----
        latents = self.model.sample(
            input_ids=prompt_ids.to(device),
            attention_mask=prompt_mask.to(device),
            uncond_input_ids=neg_ids.to(device),
            uncond_attention_mask=neg_mask.to(device),
            image_vae_latents=image_vae_latents,
            multi_video_vae_latents=multi_video_vae_latents,
            multi_image_vae_latents=multi_image_vae_latents,
            num_frames=t,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps,
            guidance_mode=guidance_mode,
            omega_V=omega_V,
            omega_I=omega_I,
            omega_TI=omega_TI,
            omega_scale=omega_scale,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            eta=eta,
            norm_threshold=norm_threshold,
            momentum=momentum,
            progress_callback=progress_callback,
        )
        self.model.to("cpu")
        torch.cuda.empty_cache()

        if not write_output:
            return None

        # ---- decode + save ----
        self.vae.to(device)
        output = _vae_decode(self.vae, latents)
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_output(output, output_path, fps=fps)
        logger.info("saved -> %s  (%d frames, %dx%d)", output_path, output.shape[0], h, w)
        return output_path
