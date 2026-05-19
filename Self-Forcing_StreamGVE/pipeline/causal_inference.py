from typing import List, Optional
import torch
import math
from tqdm import tqdm 

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


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
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def rollout_inference(
        self,
        noise: torch.Tensor,

        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        mode: Optional[str] = None,
        mode_kwargs: Optional[dict] = None,

        rollout_chunk_size: int = 21,
        rollout_overlap_block_num: int = 1,     # default 3 frame per-block
    ) -> torch.Tensor:
        '''
        rollout for long video inference
        '''
        if rollout_chunk_size < 0:
            # for testing local attn
            return self.inference(
                noise=noise,
                text_prompts=text_prompts,
                initial_latent=initial_latent,
                return_latents=return_latents,
                wo_video_decode=wo_video_decode,
                profile=profile,
                low_memory=low_memory,
                mode=mode,
                mode_kwargs=mode_kwargs,
            )

        rollout_overlap = rollout_overlap_block_num * self.num_frame_per_block 

        total_frame_num = noise.shape[1]
        ret_latent_list = []
        prev_cond = initial_latent
        start_idx = 0

        while True:            
            # skip overlap part
            if prev_cond is None:
                rollout_noise = noise[:, start_idx: start_idx + rollout_chunk_size]
            else:
                rollout_noise = noise[:, start_idx + prev_cond.shape[1]: start_idx + rollout_chunk_size]

            # inference
            _, rollout_latent = self.inference(
                noise=rollout_noise,
                text_prompts=text_prompts,
                initial_latent=prev_cond,
                profile=False,
                low_memory=low_memory,
                mode=mode,
                mode_kwargs=mode_kwargs,

                return_latents=True,
                wo_video_decode=True,
            )

            # store results
            if prev_cond is None:
                ret_latent_list.append(rollout_latent)
            else:
                ret_latent_list.append(rollout_latent[:, prev_cond.shape[1]: ])
            
            # finish, end loop
            if start_idx + rollout_chunk_size >= total_frame_num:
                break

            # index update
            start_idx += (rollout_chunk_size - rollout_overlap)

            # prepare prev_cond
            if mode is not None and 'inv' in mode:
                prev_cond = noise[:, start_idx: start_idx + rollout_overlap]
            else:
                prev_cond = rollout_latent[:, -rollout_overlap: ]

        output = torch.cat(ret_latent_list, dim=1)
        assert noise.shape == output.shape, 'noise shape: %s, but output: %s.' % (str(noise.shape), str(output.shape))

        # clean cache before decode to avoid OOM
        self.kv_cache1 = None
        self.crossattn_cache = None
        torch.cuda.empty_cache()

        if wo_video_decode:
            video = None
        else:
            if mode == 'uni-edit':
                dec_latent = output.chunk(2, dim=0)[0]
            elif mode == 'flowedit':
                dec_latent = output.chunk(4, dim=0)[0]
            else:
                dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            torch.cuda.synchronize()

        if return_latents:
            return video, output
        else:
            return video

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
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
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].

        mode & mode_kwargs: 
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
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
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
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0
            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

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

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
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

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in tqdm(enumerate(denoising_step_list), total=len(denoising_step_list), leave=False):
                
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.float32) * current_timestep

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

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if mode is not None and 'inv' in mode:
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

        # Step 4: Decode the output
        if wo_video_decode:
            video = None
        else:
            if mode == 'uni-edit':
                dec_latent = output.chunk(2, dim=0)[0]
            elif mode == 'flowedit':
                dec_latent = output.chunk(4, dim=0)[0]
            else:
                dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
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
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        self.kv_cache1 = None
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
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
        self.crossattn_cache = None
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
