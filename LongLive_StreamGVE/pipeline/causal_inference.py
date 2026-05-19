# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os
import math
from tqdm import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        # denoising_step_list: e.g., [1000, 750, 500, 250]
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            # self.scheduler.timesteps: 1000~4.9801
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            # self.denoising_step_list: tensor([1000.0000,  937.5000,  833.3333,  625.0000])
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        mode: Optional[str] = None,
        mode_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].

        mode & mode_kwargs for editing:
            None: original generation
                None kwargs 
            'uni-inv': video as 'noise'
                'alpha'
            'uni-edit': inverted latent as 'noise'
                'alpha', 'omega'
            'flowedit': video as 'noise'
                'n_avg', 'n_min', 'n_max', 'src_guidance_scale', 'trg_guidance_scale'
            'flowchef': video as 'noise'
                'cfg_scale', 'learning_rate', 'max_steps', 'optimization_steps', 'n_frames'
            'sdedit': video as 'noise'
                'strength', 'cfg_scale', 'use_sde'
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        num_inference_steps = len(self.denoising_step_list)
        if mode == 'uni-inv':
            denoising_step_list = torch.cat(
                [self.denoising_step_list, torch.zeros_like(self.denoising_step_list[-1 :])]
            ).flip(dims=[0])    # tensor([0.0000,  625.0000,  833.3333,  937.5000,  1000.0000])
            if mode_kwargs.get('alpha', 1) < 1:
                inv_steps = math.floor(mode_kwargs['alpha'] * num_inference_steps)
                skip_steps = num_inference_steps - inv_steps
                denoising_step_list = denoising_step_list[: -skip_steps]
        elif mode == 'uni-edit':
            denoising_step_list = self.denoising_step_list
            if mode_kwargs.get('alpha', 1) < 1:
                sampling_steps = math.floor(mode_kwargs['alpha'] * num_inference_steps)
                skip_steps = num_inference_steps - sampling_steps
                denoising_step_list = denoising_step_list[skip_steps: ]
        elif mode == 'flowedit':
            num_inference_steps = len(self.denoising_step_list)
            n_min, n_max = mode_kwargs['n_min'], mode_kwargs['n_max']
            denoising_step_list = self.denoising_step_list[num_inference_steps - n_max: num_inference_steps - n_min]
            # n_avg
            denoising_step_list = denoising_step_list.unsqueeze(-1).repeat_interleave(mode_kwargs['n_avg'], dim=-1).reshape(-1)
            t_last = self.denoising_step_list[num_inference_steps - n_min] if n_min > 0 else 0
        elif mode == 'sdedit':
            denoising_step_list = self.denoising_step_list * mode_kwargs['strength']
        else:
            denoising_step_list = self.denoising_step_list

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in tqdm(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]
            original_input = noisy_input
            denoised_pred = noisy_input     # init as sample

            if mode == 'flowedit':
                # n_avg, init inner loop
                inner_loop_iter = 0
                inner_avg_velocity = 0
                # store source
                x_src = noisy_input.chunk(4, dim=0)[0]
                # add noise for video latent as first step initialization
                fwd_noise = torch.randn_like(x_src).repeat_interleave(4, dim=0)
                noisy_input = self.scheduler.add_noise(
                    noisy_input.flatten(0, 1),
                    fwd_noise.flatten(0, 1),
                    denoising_step_list[0] * torch.ones(
                        [batch_size * current_num_frames], device=noisy_input.device, dtype=torch.long)
                ).unflatten(0, noisy_input.shape[:2])
            if mode == 'flowchef':
                src_latent = noisy_input
                # init as noise
                if mode_kwargs['cfg_scale'] > 1:
                    noisy_input = torch.randn_like(noisy_input.chunk(2, dim=0)[0]).repeat_interleave(2, dim=0)
                else:
                    noisy_input = torch.randn_like(noisy_input)
            if mode == 'sdedit':
                # init noise
                if mode_kwargs['cfg_scale'] > 1:
                    fwd_noise = torch.randn_like(noisy_input.chunk(2, dim=0)[0]).repeat_interleave(2, dim=0)
                else:
                    fwd_noise = torch.randn_like(noisy_input)
                # add noise
                noisy_input = self.scheduler.add_noise(
                    noisy_input.flatten(0, 1),
                    fwd_noise.flatten(0, 1),
                    denoising_step_list[0] * torch.ones(
                        [batch_size * current_num_frames], device=noisy_input.device, dtype=torch.long)
                ).unflatten(0, noisy_input.shape[:2])

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in tqdm(enumerate(denoising_step_list), total=len(denoising_step_list), leave=False):
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                # model forward
                velocity_pred, x0_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

                # get next input and result
                if mode is None:
                    if index < len(denoising_step_list) - 1:
                        next_timestep = denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            x0_pred.flatten(0, 1),
                            torch.randn_like(x0_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, x0_pred.shape[:2])
                    else:
                        # for getting real output
                        denoised_pred = x0_pred

                elif mode == 'uni-inv':
                    sigma_prev = denoising_step_list[index - 1] / 1000 if index > 0 else current_timestep / 1000
                    sigma = current_timestep / 1000
                    sigma_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else current_timestep / 1000
                    denoised_pred = denoised_pred + (sigma - sigma_prev) * velocity_pred
                    noisy_input = denoised_pred + (sigma_next - sigma) * velocity_pred

                elif mode == 'uni-edit':
                    sigma = current_timestep / 1000
                    sigma_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                    # [B, F, C, H, W]
                    v_src, v_trg = velocity_pred.chunk(2, dim=0)
                    guidance = v_trg - v_src
                    # get mask
                    mask = guidance.mean(dim=2, keepdim=True)       # [B, F, 1, H, W]
                    mask_min = mask.amin(dim=list(range(mask.ndim))[1: ], keepdim=True)
                    mask_max = mask.amax(dim=list(range(mask.ndim))[1: ], keepdim=True)
                    mask = (mask - mask_min) / (mask_max - mask_min + 1e-7)
                    # uniedit-flow
                    stride_corr = mode_kwargs.get('omega', 5.0) * (sigma_next - sigma) * (1 + mask) * guidance
                    velocity_fusion = mask * v_trg + (1 - mask) * v_src
                    stride_corr = torch.cat([stride_corr, stride_corr], dim=0)
                    velocity_fusion = torch.cat([velocity_fusion, velocity_fusion], dim=0)
                    # forward
                    denoised_pred = denoised_pred + stride_corr + (sigma_next - sigma) * velocity_fusion
                    noisy_input = denoised_pred

                elif mode == 'flowedit':
                    # velocity update
                    v_src_uncond, v_src_cond, v_trg_uncond, v_trg_cond = velocity_pred.chunk(4, dim=0)
                    v_src = v_src_uncond + mode_kwargs['src_guidance_scale'] * (v_src_cond - v_src_uncond)
                    v_trg = v_trg_uncond + mode_kwargs['trg_guidance_scale'] * (v_trg_cond - v_trg_uncond)
                    inner_loop_iter += 1
                    inner_avg_velocity += (1 / mode_kwargs['n_avg']) * (v_trg - v_src)
                    # edit step
                    t_i = current_timestep / 1000
                    if inner_loop_iter == mode_kwargs['n_avg']:
                        t_im1 = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else t_last
                        denoised_pred = denoised_pred + (t_im1 - t_i) * inner_avg_velocity
                        inner_loop_iter = 0
                        inner_avg_velocity = 0
                        t_i = t_im1     # for next step
                    # prepare for subsequent steps
                    fwd_noise = torch.randn_like(x_src)
                    zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                    zt_tar = denoised_pred.chunk(4, dim=0)[0] + zt_src - x_src
                    noisy_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])

                elif mode == 'flowchef':
                    if mode_kwargs['cfg_scale'] > 1:
                        v_src, v_trg = velocity_pred.chunk(2, dim=0)
                        velocity = v_src + (v_trg - v_src) * mode_kwargs['cfg_scale']
                    else:
                        velocity = velocity_pred
                    sigma = current_timestep / 1000
                    sigma_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                    sample = noisy_input
                    if index < mode_kwargs['max_steps']:
                        opt_latents = sample.detach().clone()
                        with torch.enable_grad():
                            opt_latents = opt_latents.detach().requires_grad_()
                            opt_latents = torch.autograd.Variable(opt_latents, requires_grad=True)

                            for _ in range(mode_kwargs['optimization_steps']):
                                latents_p = opt_latents - sigma * velocity
                                loss = (
                                    1000*torch.nn.functional.mse_loss(latents_p, src_latent, reduction='none')
                                ).mean() * mode_kwargs['n_frames']

                                grad = torch.autograd.grad(loss, opt_latents)[0]
                                # grad = torch.clamp(grad, -0.5, 0.5)
                                opt_latents = opt_latents - mode_kwargs['learning_rate'] * grad

                        sample = opt_latents.detach().clone()
                    denoised_pred = sample - sigma * velocity
                    noisy_input = denoised_pred + sigma_next * velocity

                elif mode == 'sdedit':
                    sigma = current_timestep / 1000
                    sigma_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                    if mode_kwargs['cfg_scale'] > 1:
                        v_src, v_trg = velocity_pred.chunk(2, dim=0)
                        velocity = v_src + (v_trg - v_src) * mode_kwargs['cfg_scale']
                    else:
                        velocity = velocity_pred
                    denoised_pred = noisy_input - sigma * velocity
                    if mode_kwargs['use_sde']:
                        noisy_input = (1 - sigma_next) * denoised_pred + sigma_next * torch.randn_like(velocity)
                    else:
                        noisy_input = denoised_pred + sigma_next * velocity

            # Step 2.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if mode is not None and  'inv' in mode:
                clean_data = original_input
            else:
                clean_data = denoised_pred
            self.generator(
                noisy_image_or_video=clean_data,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        if wo_video_decode:
            video = None
        else:
            if mode == 'uni-edit':
                dec_latent = output.chunk(2, dim=0)[0]
            elif mode == 'flowedit':
                dec_latent = output.chunk(4, dim=0)[0]
            else:
                dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent.to(noise.device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass