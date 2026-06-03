# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

"""Wan2.2 dual-expert diffusion sampler with APG / chained guidance."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from .scheduler import FlowMatchScheduler
from .transformer_wan import WanTransformer3DModel


# --------------------------------------------------------------------------- #
# Adaptive Projected Guidance (https://arxiv.org/pdf/2410.02416)
# --------------------------------------------------------------------------- #
class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average


def _normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
    """Project `diff` onto / off `base_pred` and recombine with weight `eta`."""
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), base_pred.double()
    v1 = F.normalize(v1, dim=[-1, -2, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.to(diff.dtype), v0_orthogonal.to(diff.dtype)
    return diff_orthogonal + eta * diff_parallel


def normalized_guidance(
    pred_cond, pred_uncond, guidance_scale, momentum_buffer=None, eta=1.0, norm_threshold=0.0
):
    """Single-condition APG."""
    nd = _normalize_diff(pred_cond - pred_uncond, pred_cond, momentum_buffer, eta, norm_threshold)
    return pred_uncond + guidance_scale * nd


def normalized_guidance_chain(pred_uncond, preds, scales, momentum_buffers, eta, norm_thresholds):
    """Chained APG: each condition's diff is taken against the previous one."""
    bases = [pred_uncond] + list(preds)
    result = pred_uncond
    for i, cond in enumerate(preds):
        nd = _normalize_diff(cond - bases[i], cond, momentum_buffers[i], eta, norm_thresholds[i])
        result = result + scales[i] * nd
    return result


_PACK = "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)"
_UNPACK = "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)"


def _to_spatial(x, shape):
    return rearrange(x, _PACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def _to_packed(x, shape):
    return rearrange(x, _UNPACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


class GEN_Wanx22(nn.Module):
    """Dual-expert (high-noise / low-noise) Wan2.2 transformer with guidance."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.switch_dit_boundary = config.switch_dit_boundary
        self.model_id_or_path = config.wan22_base

        common = dict(
            use_src_id_rotary_emb=config.use_src_id_rotary_emb,
            torch_dtype=torch.bfloat16,
        )
        # from_pretrained loads the Wan2.2 base transformer (bf16, with the
        # precision-sensitive modules kept in fp32); the Bernini checkpoint is
        # applied afterwards by bernini.weights.load_weights.
        if config.skip_transformer_1:
            self.transformer = None
        else:
            self.transformer = WanTransformer3DModel.from_pretrained(
                self.model_id_or_path, subfolder="transformer", **common
            )
            self.config.text_dim = self.transformer.config.text_dim
            self.rope = self.transformer.rope
        if config.skip_transformer_2:
            self.transformer_2 = None
        else:
            self.transformer_2 = WanTransformer3DModel.from_pretrained(
                self.model_id_or_path, subfolder="transformer_2", **common
            )
            self.config.text_dim = self.transformer_2.config.text_dim
            self.rope = self.transformer_2.rope

        self.use_unipc = config.use_unipc
        if self.use_unipc:
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                self.model_id_or_path, subfolder="scheduler", flow_shift=config.shift
            )
        else:
            self.scheduler = FlowMatchScheduler(shift=config.shift, sigma_min=0.0, extra_one_step=False)

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8

    def shared_step(self, model_id, noisy_latents, timesteps, cond_embeds, rotary_embs,
                    batch_vae_seqlen, batch_text_seqlen):
        cur_transformer = self.transformer if model_id == "transformer_1" else self.transformer_2
        assert cur_transformer is not None
        return cur_transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=cond_embeds,
            rotary_emb=rotary_embs,
            batch_image_vae_seqlen=batch_vae_seqlen,
            text_features_length=batch_text_seqlen,
        ).sample

    def _apg_sigma(self, t_idx: int):
        """Noise level at the current step, for converting v-pred to x-pred."""
        if hasattr(self.scheduler, "step_index"):
            idx = 0 if self.scheduler.step_index is None else self.scheduler.step_index
            return self.scheduler.sigmas[idx]
        return self.scheduler.sigmas[t_idx]

    @torch.no_grad()
    def sample(
        self,
        prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_prompt_embeds=None,
        uncond_embeds_t2=None,
        num_frames=1,
        width=832,
        height=480,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_inference_steps=50,
        guidance_mode="rv2v",
        omega_V=3.0,
        omega_I=3.0,
        omega_TI=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        seed=42,
        device="cuda",
        eta=1.0,
        norm_threshold=(50.0, 50.0),
        momentum=0.0,
        progress_callback=None,
    ):
        """Run guided sampling and return the predicted VAE latent `[B,C,T,H,W]`.

        guidance_mode:
          - ``rv2v``      : reference + video editing (chained, 4 forwards)
          - ``v2v``       : video editing, plain CFG (2 forwards)
          - ``v2v_chain`` : video editing, chained CFG (3 forwards)
          - ``t2v``       : text-to-video, plain CFG (2 forwards)
          - ``r2v_apg``   : reference-to-video, APG chained (3 forwards)
          - ``v2v_apg``   : video editing, single-condition APG (2 forwards)
          - ``t2v_apg``   : text-to-video, single-condition APG (2 forwards)
        """
        if self.use_unipc:
            self.scheduler.set_timesteps(num_inference_steps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)

        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        prompt_embeds_t1 = prompt_embeds
        if prompt_embeds_t2 is None:
            prompt_embeds_t2 = prompt_embeds
        uncond_embeds_t1 = uncond_prompt_embeds
        if uncond_embeds_t2 is None:
            uncond_embeds_t2 = uncond_prompt_embeds

        timesteps = self.scheduler.timesteps.to(device)
        boundary_timestep = self.switch_dit_boundary * self.scheduler.num_train_timesteps

        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = randn_tensor(shape, device=device, dtype=torch.float32, generator=gen)
        noisy_vae_latent = rearrange(noise, "b c t (h ph) (w pw) -> b (t h w) (ph pw c)", ph=2, pw=2)
        noisy_vae_latent = noisy_vae_latent.to(device)

        self.transformer.to(device)
        if self.transformer_2 is not None:
            self.transformer_2.to("cpu")
        torch.cuda.empty_cache()
        switched = False

        # APG momentum buffers / per-condition norm thresholds.
        if guidance_mode == "r2v_apg":
            if isinstance(norm_threshold, (int, float)):
                norm_threshold = [norm_threshold, norm_threshold]
            elif len(norm_threshold) == 1:
                norm_threshold = [norm_threshold[0], norm_threshold[0]]
            momentum_buffer1 = MomentumBuffer(momentum)
            momentum_buffer2 = MomentumBuffer(momentum)
        elif guidance_mode in ("v2v_apg", "t2v_apg"):
            momentum_buffer = MomentumBuffer(momentum)
        nt0 = norm_threshold[0] if isinstance(norm_threshold, (list, tuple)) else norm_threshold

        progress_bar = tqdm(timesteps)
        for t_idx, t in enumerate(timesteps):
            model_id = "transformer_1" if t >= boundary_timestep else "transformer_2"
            cond_text = prompt_embeds_t1 if t >= boundary_timestep else prompt_embeds_t2
            uncond_text = uncond_embeds_t1 if t >= boundary_timestep else uncond_embeds_t2

            if t < boundary_timestep and not switched and self.transformer_2 is not None:
                self.transformer.to("cpu")
                torch.cuda.empty_cache()
                self.transformer_2.to(device)
                switched = True
                omega_V *= omega_scale
                omega_I *= omega_scale
                omega_TI *= omega_scale

            cur_transformer = self.transformer_2 if switched else self.transformer

            # ----------------------------------------------------------------
            # Build conditioning combos. Each combo = condition tokens + the
            # shared noisy target latent (source_id 0).
            #   V  : video only            I : reference image(s) only
            #   VI : video + image(s)      none : no conditioning
            # ----------------------------------------------------------------
            v_latents, v_rotary, v_masks, v_len = [], [], [], 0
            i_latents, i_rotary, i_masks, i_len = [], [], [], 0
            vi_latents, vi_rotary, vi_masks, vi_len = [], [], [], 0
            sid = 1       # global source_id for the VI combo
            sid_img = 1   # source_id for the image-only combo

            target_video_latents = []
            if multi_video_vae_latents is not None:
                if isinstance(multi_video_vae_latents, torch.Tensor):
                    target_video_latents = [multi_video_vae_latents]
                else:
                    target_video_latents = multi_video_vae_latents

            for idx, video_latent in enumerate(target_video_latents):
                cur_latent, rotary_emb = cur_transformer.patch_vae_latent(
                    video_latent.to(dtype=cur_transformer.dtype), source_id=sid
                )
                sid += 1
                mask = torch.zeros(cur_latent.shape[1], device=device, dtype=torch.bool)
                if idx == 0:  # only the first video joins the V combo
                    v_latents.append(cur_latent)
                    v_rotary.append(rotary_emb)
                    v_masks.append(mask)
                    v_len += cur_latent.shape[1]
                vi_latents.append(cur_latent)
                vi_rotary.append(rotary_emb)
                vi_masks.append(mask)
                vi_len += cur_latent.shape[1]

            def _add_image(img_vae):
                nonlocal sid, sid_img, vi_len, i_len
                cur_latent, rotary_emb = cur_transformer.patch_vae_latent(
                    img_vae.to(dtype=cur_transformer.dtype), source_id=sid
                )
                sid += 1
                vi_latents.append(cur_latent)
                vi_rotary.append(rotary_emb)
                vi_masks.append(torch.zeros(cur_latent.shape[1], device=device, dtype=torch.bool))
                vi_len += cur_latent.shape[1]

                cur_latent_i, rotary_emb_i = cur_transformer.patch_vae_latent(
                    img_vae.to(dtype=cur_transformer.dtype), source_id=sid_img
                )
                sid_img += 1
                i_latents.append(cur_latent_i)
                i_rotary.append(rotary_emb_i)
                i_masks.append(torch.zeros(cur_latent_i.shape[1], device=device, dtype=torch.bool))
                i_len += cur_latent_i.shape[1]

            if image_vae_latents is not None:
                for idx in range(image_vae_latents.shape[2]):
                    _add_image(image_vae_latents[:, :, idx : idx + 1, :, :])
            if multi_image_vae_latents is not None:
                for img_vae in multi_image_vae_latents:
                    _add_image(img_vae)

            # Noisy target latent, shared across all combos (source_id 0).
            unpacked_noisy_latent = _to_spatial(noisy_vae_latent, shape).to(cur_transformer.dtype)
            noisy_latent, noisy_rotary = cur_transformer.patch_vae_latent(unpacked_noisy_latent, source_id=0)
            noisy_len = noisy_latent.shape[1]
            noisy_mask = torch.ones(noisy_len, device=device, dtype=torch.bool)

            def _assemble(cond_lats, cond_rots, cond_msks, cond_len):
                return (
                    torch.cat(cond_lats + [noisy_latent], dim=1).to(cur_transformer.dtype),
                    torch.cat(cond_rots + [noisy_rotary], dim=2),
                    torch.cat(cond_msks + [noisy_mask], dim=0),
                    cond_len + noisy_len,
                )

            none_inp, none_rot, none_msk, none_total = _assemble([], [], [], 0)
            v_inp, v_rot, v_msk, v_total = _assemble(v_latents, v_rotary, v_masks, v_len)
            i_inp, i_rot, i_msk, i_total = _assemble(i_latents, i_rotary, i_masks, i_len)
            vi_inp, vi_rot, vi_msk, vi_total = _assemble(vi_latents, vi_rotary, vi_masks, vi_len)

            timestep = t.expand(1)

            def _fwd(lat_inp, rot, msk, total, text_emb):
                pred = self.shared_step(
                    model_id=model_id,
                    noisy_latents=lat_inp,
                    timesteps=timestep,
                    cond_embeds=text_emb,
                    rotary_embs=rot,
                    batch_vae_seqlen=[total],
                    batch_text_seqlen=[text_emb.shape[1]],
                )
                return pred[:, msk, :]

            # ----------------------------------------------------------------
            # Guidance.
            # ----------------------------------------------------------------
            if guidance_mode == "rv2v":
                # ε̂ = ε_∅ + ω_V(ε_V-ε_∅) + ω_I(ε_VI-ε_V) + ω_TI(ε_VTI-ε_VI)
                eps_uncond = _fwd(none_inp, none_rot, none_msk, none_total, uncond_text)
                eps_V = _fwd(v_inp, v_rot, v_msk, v_total, uncond_text)
                eps_VI = _fwd(vi_inp, vi_rot, vi_msk, vi_total, uncond_text)
                eps_VTI = _fwd(vi_inp, vi_rot, vi_msk, vi_total, cond_text)
                noise_pred = (
                    eps_uncond
                    + omega_V * (eps_V - eps_uncond)
                    + omega_I * (eps_VI - eps_V)
                    + omega_TI * (eps_VTI - eps_VI)
                )

            elif guidance_mode == "v2v":
                # Video editing, plain CFG over text with the V+I condition
                # fixed: ε̂ = ε_VI + ω_TI(ε_VTI - ε_VI)
                eps_uncond = _fwd(vi_inp, vi_rot, vi_msk, vi_total, uncond_text)
                eps_VTI = _fwd(vi_inp, vi_rot, vi_msk, vi_total, cond_text)
                noise_pred = eps_uncond + omega_TI * (eps_VTI - eps_uncond)

            elif guidance_mode == "v2v_chain":
                # Video editing, chained CFG: ε̂ = ε_∅ + ω_V(ε_V-ε_∅) + ω_TI(ε_VTI-ε_V)
                eps_uncond = _fwd(none_inp, none_rot, none_msk, none_total, uncond_text)
                eps_V = _fwd(v_inp, v_rot, v_msk, v_total, uncond_text)
                eps_VTI = _fwd(vi_inp, vi_rot, vi_msk, vi_total, cond_text)
                noise_pred = (
                    eps_uncond
                    + omega_V * (eps_V - eps_uncond)
                    + omega_TI * (eps_VTI - eps_V)
                )

            elif guidance_mode == "t2v":
                # Text-to-video, plain CFG: ε̂ = ε_∅ + ω_TI(ε_T-ε_∅)
                eps_uncond = _fwd(none_inp, none_rot, none_msk, none_total, uncond_text)
                eps_T = _fwd(none_inp, none_rot, none_msk, none_total, cond_text)
                noise_pred = eps_uncond + omega_TI * (eps_T - eps_uncond)

            elif guidance_mode == "r2v_apg":
                # Reference-to-video: no source video. Chained APG over ∅ / I / TI.
                eps_uncond = _fwd(none_inp, none_rot, none_msk, none_total, uncond_text)
                eps_I = _fwd(i_inp, i_rot, i_msk, i_total, uncond_text)
                eps_TI = _fwd(i_inp, i_rot, i_msk, i_total, cond_text)
                sigma_apg = self._apg_sigma(t_idx)
                noisy_r = _to_spatial(noisy_vae_latent, shape)
                eps_uncond_r = noisy_r - sigma_apg * _to_spatial(eps_uncond, shape)
                eps_I_r = noisy_r - sigma_apg * _to_spatial(eps_I, shape)
                eps_TI_r = noisy_r - sigma_apg * _to_spatial(eps_TI, shape)
                x_guided = normalized_guidance_chain(
                    pred_uncond=eps_uncond_r,
                    preds=[eps_I_r, eps_TI_r],
                    scales=[omega_I, omega_TI],
                    momentum_buffers=[momentum_buffer1, momentum_buffer2],
                    eta=eta,
                    norm_thresholds=norm_threshold,
                )
                noise_pred = _to_packed((noisy_r - x_guided) / sigma_apg, shape)

            elif guidance_mode == "v2v_apg":
                # Video editing: single-condition APG between ∅ and VTI.
                eps_uncond = _fwd(vi_inp, vi_rot, vi_msk, vi_total, uncond_text)
                eps_VTI = _fwd(vi_inp, vi_rot, vi_msk, vi_total, cond_text)
                sigma_apg = self._apg_sigma(t_idx)
                noisy_r = _to_spatial(noisy_vae_latent, shape)
                eps_uncond_r = noisy_r - sigma_apg * _to_spatial(eps_uncond, shape)
                eps_VTI_r = noisy_r - sigma_apg * _to_spatial(eps_VTI, shape)
                x_guided = normalized_guidance(
                    pred_cond=eps_VTI_r,
                    pred_uncond=eps_uncond_r,
                    guidance_scale=omega_TI,
                    momentum_buffer=momentum_buffer,
                    eta=eta,
                    norm_threshold=nt0,
                )
                noise_pred = _to_packed((noisy_r - x_guided) / sigma_apg, shape)

            elif guidance_mode == "t2v_apg":
                # Text-to-video: single-condition APG between ∅ and T.
                eps_uncond = _fwd(none_inp, none_rot, none_msk, none_total, uncond_text)
                eps_T = _fwd(none_inp, none_rot, none_msk, none_total, cond_text)
                sigma_apg = self._apg_sigma(t_idx)
                noisy_r = _to_spatial(noisy_vae_latent, shape)
                eps_uncond_r = noisy_r - sigma_apg * _to_spatial(eps_uncond, shape)
                eps_T_r = noisy_r - sigma_apg * _to_spatial(eps_T, shape)
                x_guided = normalized_guidance(
                    pred_cond=eps_T_r,
                    pred_uncond=eps_uncond_r,
                    guidance_scale=omega_TI,
                    momentum_buffer=momentum_buffer,
                    eta=eta,
                    norm_threshold=nt0,
                )
                noise_pred = _to_packed((noisy_r - x_guided) / sigma_apg, shape)

            else:
                raise ValueError(
                    f"Unknown guidance_mode='{guidance_mode}'. Expected one of: "
                    f"rv2v, v2v, v2v_chain, t2v, r2v_apg, v2v_apg, t2v_apg."
                )

            if isinstance(self.scheduler, FlowMatchScheduler):
                noisy_vae_latent = self.scheduler.step(noise_pred, t, noisy_vae_latent, return_dict=False)
            else:
                noisy_vae_latent = self.scheduler.step(noise_pred, t, noisy_vae_latent, return_dict=False)[0]

            progress_bar.update(1)
            if progress_callback is not None:
                progress_callback(t_idx + 1, len(timesteps))

        return _to_spatial(noisy_vae_latent, shape)
