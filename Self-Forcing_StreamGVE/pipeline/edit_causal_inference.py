from typing import List, Optional, Iterable
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm 

import os
import numpy as np
from PIL import Image

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .utils import find_phrase_token_indices

class EditCausalInferencePipeline(torch.nn.Module):
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

        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _check_prompts(self, *args):
        ret_list = []
        for itm in args:
            if isinstance(itm, str):
                itm = [itm]
            ret_list.append(itm)
        return ret_list


    def rollout_inference(
        self,
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,  
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,

        rollout_chunk_size: int = 21,
        rollout_overlap_block_num: int = 1,
    ) -> torch.Tensor:
        if rollout_chunk_size < 0:
            # for testing local attn
            return self.inference(
                src_video=src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,
                return_latents=return_latents,
                wo_video_decode=wo_video_decode,
                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame,
                triple_first_frame=triple_first_frame,
                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,
            )

        rollout_overlap = rollout_overlap_block_num * self.num_frame_per_block 

        total_frame_num = src_video.shape[1]
        ret_latent_list = []
        start_idx = 0

        while True:
            chunk_right_idx = start_idx + rollout_chunk_size
            if (start_idx == 0) and (independent_first_frame or triple_first_frame):
                # provide kv_cache space for image condition
                chunk_right_idx -= self.num_frame_per_block

            if start_idx == 0:
                rollout_src_video = src_video[:, start_idx: chunk_right_idx]
            else:
                rollout_src_video = src_video[:, start_idx + rollout_overlap: chunk_right_idx]

            # inference
            _, rollout_latent = self.inference(
                src_video=rollout_src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,

                return_latents=True,
                wo_video_decode=True,

                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame if start_idx == 0 else False,
                triple_first_frame=triple_first_frame if start_idx == 0 else False,
                
                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,
            )

            # store results
            if start_idx == 0:
                ret_latent_list.append(rollout_latent)
            else:
                ret_latent_list.append(rollout_latent[:, rollout_overlap: ])
            
            # finish, end loop
            if chunk_right_idx >= total_frame_num:
                break

            # index update
            start_idx = chunk_right_idx - rollout_overlap

            # prepare prev_cond
            src_initial_latent = rollout_src_video[:, -rollout_overlap: ]
            trg_initial_latent = rollout_latent[:, -rollout_overlap: ]

        output = torch.cat(ret_latent_list, dim=1)
        assert src_video.shape == output.shape, 'noise shape: %s, but output: %s.' % (str(src_video.shape), str(output.shape))

        # clean cache before decode to avoid OOM
        torch.cuda.empty_cache()

        if wo_video_decode:
            video = None
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
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,  
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,
    ) -> torch.Tensor:
        assert not (independent_first_frame and triple_first_frame)
        independent_first_frame = independent_first_frame or self.independent_first_frame
        
        batch_size, num_frames, num_channels, height, width = src_video.shape
        if not independent_first_frame or (independent_first_frame and trg_initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = trg_initial_latent.shape[1] if trg_initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        src_prompts, trg_prompts, src_trigger_words, trg_trigger_words = self._check_prompts(
            src_prompts, trg_prompts, src_trigger_words, trg_trigger_words
        )
        conditional_dict = self.text_encoder(
            text_prompts=src_prompts + trg_prompts
        )
        src_conditional_dict = self.text_encoder(
            text_prompts=src_prompts
        )
        trg_conditional_dict = self.text_encoder(
            text_prompts=trg_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=src_video.device,
            dtype=src_video.dtype
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

        # Step 1: Initialize KV cache, trg_fg_mask cache, and crossattn cache
        kv_cache_src = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        kv_cache_trg = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        trg_fg_mask_cache = self._initialize_trg_fg_mask_cache(
            batch_size=batch_size,
            device=src_video.device
        )
        crossattn_cache_src = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_trg = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_dual = self._initialize_crossattn_cache(
            batch_size=batch_size * 2,
            dtype=src_video.dtype,
            device=src_video.device
        )
        # Initialize some helper
        self._initialize_noise_statistics(reuse_noise_temporal_mean)

        # get trigger token indices
        trans_tokenizer = self.text_encoder.tokenizer.tokenizer
        tok_src = find_phrase_token_indices(trans_tokenizer, src_prompts, src_trigger_words)
        tok_trg = find_phrase_token_indices(trans_tokenizer, trg_prompts, trg_trigger_words)
        print(tok_src, tok_trg)

        # Step 2: Cache context feature
        current_start_frame = 0
        if trg_initial_latent is not None:
            # obtain both kv_cache and mask of both src and trg
            timestep = torch.zeros([batch_size, 1], device=src_video.device, dtype=torch.int64)
            if independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                slice_list = [(0, 1)] + [
                    (1 + idx * self.num_frame_per_block, 1 + (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block
                slice_list = [
                    (idx * self.num_frame_per_block, (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]

            for left, right in slice_list:
                context_timestep = torch.ones(
                    [batch_size, right - left], device=src_video.device, dtype=torch.float32
                ) * self.args.context_noise
                #✨ src and src mask
                current_src_ref_latents = src_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
                self.generator(
                    noisy_image_or_video=current_src_ref_latents,
                    conditional_dict=src_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_src,
                    crossattn_cache=crossattn_cache_src,
                    current_start=left * self.frame_seq_length,
                )
                _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
                #✨ trg and trg mask
                current_trg_ref_latents = trg_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
                self.generator(
                    noisy_image_or_video=current_trg_ref_latents,
                    conditional_dict=trg_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_trg,
                    crossattn_cache=crossattn_cache_trg,
                    current_start=left * self.frame_seq_length,
                )
                _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)

                #✨ src & trg union
                current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
                self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)

                output[:, left: right] = current_trg_ref_latents
                current_start_frame = right

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        denoising_step_list = self.denoising_step_list
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if independent_first_frame and trg_initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm(all_num_frames):
            if profile:
                block_start.record()

            src_input = src_video[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            denoised_pred = src_input

            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=src_video.device,
                dtype=torch.float32
            ) * self.args.context_noise
            
            # obtain currently inprocessed kv_cache for dual branch
            shared_dict_dual = dict()
            kv_cache_dual = self._concat_kv_cache(kv_cache_src, kv_cache_trg, shared_dict=shared_dict_dual)

            #✨ forward clean source video to get source mask, and store into kv_cache
            self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
            self.generator(
                noisy_image_or_video=src_input,
                conditional_dict=src_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_src,
                crossattn_cache=crossattn_cache_src,
                current_start=current_start_frame * self.frame_seq_length,
            )
            _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
            # inject to kv_cache
            self._inject_masks_to_kv_cache(
                kv_cache_dual, trg_fg_mask_cache, src_fg_mask_bin, 
            )
            src_fg_mask_map = self._mask_reshape(
                src_fg_mask_bin, size=(current_num_frames, height, width)
            )
            inloop_trg_fg_mask = src_fg_mask_bin
            
            # Step 3.1: Spatial denoising loop
            noisy_pred_input = None
            for index, current_timestep in tqdm(enumerate(denoising_step_list), total=len(denoising_step_list), leave=False):
                
                # set current timestep
                timestep = torch.ones(
                    [batch_size * 2, current_num_frames],
                    device=src_video.device,
                    dtype=torch.float32
                ) * current_timestep
                timestep_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                shared_dict_dual['current_timestep_next'] = float(timestep_next)
                shared_dict_dual['current_timestep'] = float(current_timestep / 1000)
                shared_dict_dual['current_timestep_index'] = index
                shared_dict_dual['total_timestep'] = len(denoising_step_list)
                shared_dict_dual['blend_power'] = blend_power

                # use previous statistics on noise
                fwd_noise = torch.randn_like(src_input)
                fwd_noise = self._reuse_noise_statistics(fwd_noise, index, fg_mask=src_fg_mask_map)
                fwd_trg_noise = fwd_noise

                # update mask with trg mask at t^inj=0.5
                if index == len(denoising_step_list) // 2:
                    self._register_crossattn_mask_gatherer(crossattn_cache_dual, tok_src + tok_trg, layers=mask_layers, fg_scale=fg_scale)

                if fg_boost_factor != 1.0:
                    self._register_crossattn_enhancement(
                        crossattn_cache_dual, tok_src + tok_trg, 
                        layers=enhance_layers, fg_boost_factor=fg_boost_factor,
                        current_src_fg_mask=inloop_trg_fg_mask,
                    )

                # add noise to both source video and generating video
                noisy_src_input = self.scheduler.add_noise(
                    src_input.flatten(0, 1),
                    fwd_noise.flatten(0, 1),
                    timestep[: batch_size],
                ).unflatten(0, src_input.shape[:2])
                noisy_pred_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    fwd_trg_noise.flatten(0, 1),
                    timestep[batch_size: ],
                ).unflatten(0, denoised_pred.shape[:2])
                noisy_input = torch.cat([noisy_src_input, noisy_pred_input], dim=0)

                # model forward
                velocity_pred, _ = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache_dual,
                    crossattn_cache=crossattn_cache_dual,
                    current_start=current_start_frame * self.frame_seq_length
                )
                # for getting real output
                t_i = current_timestep / 1000
                v_src, v_trg = velocity_pred.chunk(2, dim=0)
                v_gt = fwd_noise - src_input
                
                #✨ source-oriented guidance
                fg_mask = (v_trg - v_src).abs().mean(dim=2, keepdim=True)     # [B, F, 1, H, W]
                data_dims = list(range(fg_mask.ndim))[1: ]
                fg_mask = (fg_mask - fg_mask.amin(dim=data_dims, keepdim=True)) / \
                    (fg_mask.amax(dim=data_dims, keepdim=True) - fg_mask.amin(dim=data_dims, keepdim=True) + 1e-7)
                bg_mask = 1 - fg_mask
                v_t = v_trg + bg_mask * (v_gt - v_src)
                denoised_pred = noisy_pred_input - t_i * v_t

                #✨ target mask grounding
                if index == len(denoising_step_list) // 2:
                    _, inloop_src_trg_fg_mask_bin, mask_soft_vis, mask_bin_vis = self._aggregate_crossattn_mask(
                        crossattn_cache_dual, size=(current_num_frames, height, width), scale_factor=16
                    )
                    inloop_trg_fg_mask_bin = inloop_src_trg_fg_mask_bin.chunk(2, dim=0)[1]
                    # inject union of origin src and in-processing trg masks to kv_cache
                    inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin
                    self._inject_masks_to_kv_cache(
                        kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask, 
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            del kv_cache_dual
            self._kv_cache_to(kv_cache_trg, 'cuda', low_memory)
            self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=trg_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_trg,
                crossattn_cache=crossattn_cache_trg,
                current_start=current_start_frame * self.frame_seq_length,
            )
            #✨ store clean target kv cache, and obtain clean target mask
            _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)
            current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
            self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
            self._kv_cache_to(kv_cache_trg, 'cpu', low_memory)

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
        if independent_first_frame:
            output = output[:, 1: ]
        if triple_first_frame:
            output = output[:, 3: ]
        if wo_video_decode:
            video = None
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

        return kv_cache1  # always store the clean cache

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
        return crossattn_cache


    def _initialize_trg_fg_mask_cache(self, batch_size, device):
        '''
        ✨ initialize target mask as ones
        '''
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        trg_fg_mask_cache = {
            "trg_fg_mask": torch.ones([batch_size, kv_cache_size], dtype=torch.bool, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        }
        return trg_fg_mask_cache

    def _update_trg_fg_mask_cache(self, trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg):
        '''
        ✨ update trg_fg_mask similar to kv cache update
        '''
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        kv_cache_size = trg_fg_mask_cache["trg_fg_mask"].shape[1]
        num_new_tokens = current_trg_fg_mask.shape[1]
        assert num_new_tokens == kv_cache_trg[0]["num_new_tokens"], '%d != %d' % (num_new_tokens, kv_cache_trg[0]["num_new_tokens"])
        if self.local_attn_size != -1 and (current_end > trg_fg_mask_cache["global_end_index"].item()) and (
                num_new_tokens + trg_fg_mask_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + trg_fg_mask_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = trg_fg_mask_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens: sink_tokens + num_rolled_tokens] = \
                trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens + num_evicted_tokens: sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - \
                trg_fg_mask_cache["global_end_index"].item() - num_evicted_tokens
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - trg_fg_mask_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        trg_fg_mask_cache["trg_fg_mask"][:, local_start_index:local_end_index] = current_trg_fg_mask
        trg_fg_mask_cache["global_end_index"].fill_(current_end)
        trg_fg_mask_cache["local_end_index"].fill_(local_end_index)


    def _concat_kv_cache(self, kvc_1, kvc_2, index_select=-1, shared_dict=None):
        '''
        ✨ concat source and target kv cache at batch dim for dual branch sampling
        '''
        kv_cache1 = []
        if index_select == -1:
            index_kvc = kvc_2
        else:
            index_kvc = kvc_1
        for b_idx in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.cat((kvc_1[b_idx]["k"], kvc_2[b_idx]["k"]), dim=0).clone(),
                "v": torch.cat((kvc_1[b_idx]["v"], kvc_2[b_idx]["v"]), dim=0).clone(),
                "global_end_index": index_kvc[b_idx]["global_end_index"].clone(),
                "local_end_index": index_kvc[b_idx]["local_end_index"].clone(),
                "shared_dict": shared_dict,
            })
        return kv_cache1

    def _append_clean_src_kv_cache(self, kvc_dual, kvc_src):
        '''
        ✨ add clean src kv cache to dual cache dict
        '''
        for b_idx in range(self.num_transformer_blocks):
            kvc_dual[b_idx].update({
                'k_src_clean': kvc_src[b_idx]['k'],
                'v_src_clean': kvc_src[b_idx]['v'],
            })

    def _inject_masks_to_kv_cache(
        self, kv_cache, 
        trg_fg_mask_cache=None, current_src_fg_mask=None,
    ):
        '''
        ✨
        trg_fg_mask: [B, kv_cache_size], previous chunks' foreground mask.
        current_src_fg_mask: [B, lq], current chunk's foreground mask.
        '''
        for b_idx in range(self.num_transformer_blocks):
            kv_cache[b_idx].update({
                "trg_fg_mask": trg_fg_mask_cache['trg_fg_mask'],
                "current_src_fg_mask": current_src_fg_mask,
            })
    
    def _kv_cache_to(self, kvc, device, low_memory):
        if not low_memory:
            return
        for itm in kvc:
            for k, v in itm.items():
                if isinstance(v, torch.Tensor):
                    v.to(device)


    def _register_crossattn_enhancement(self, crossattn_cache, fg_indices, fg_boost_factor=1.0, layers=range(30), current_src_fg_mask=None):
        '''
        ✨ default [src, trg] for multiple batches
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_boost_factor"] = fg_boost_factor
            crossattn_cache[l_idx]["current_src_fg_mask"] = current_src_fg_mask
            crossattn_cache[l_idx]["apply_enhance"] = True
        
    def _register_crossattn_mask_gatherer(self, crossattn_cache, fg_indices, fg_scale=1.0, layers=range(20)):
        '''
        ✨ fg_indices will be poped in blocks to avoid repeating
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_scale"] = fg_scale
            crossattn_cache[l_idx]["obtain_mask"] = True

    def _aggregate_crossattn_mask(self, crossattn_cache, size=None, patch=(1, 2, 2), scale_factor=1):
        '''
        ✨
        size: (Ttok, Htok, Wtok), for visualization only. \\
        patch: patchify kernel size. \\
        
        crossattn_cache[l_idx]["fg_mask_soft"]: [B, Lq, 1, 1] \\
        return:
            mask_soft, mask_bin: [B, Lq]
            mask_soft_vis, mask_bin_vis: [B, Ttok, Htok, Wtok]
        '''
        total_mask = 0
        account = 0
        for l_idx in range(self.num_transformer_blocks):
            if "fg_mask_soft" in crossattn_cache[l_idx]:
                total_mask += crossattn_cache[l_idx]["fg_mask_soft"].squeeze(-1).squeeze(-1)
                account += 1
        mask_soft = total_mask / account
        mask_bin = mask_soft > 0
        if size is None:
            mask_soft_vis = None
            mask_bin_vis = None
        else:
            view_size = (total_mask.size(0), *map(lambda s, p: s // p, size, patch))
            mask_soft_vis = mask_soft.view(view_size)
            mask_bin_vis = mask_bin.view(view_size)
            if scale_factor != 1:
                mask_soft_vis = F.interpolate(mask_soft_vis, scale_factor=scale_factor).to(mask_soft_vis)
                mask_bin_vis = F.interpolate(mask_bin_vis.float(), scale_factor=scale_factor) > 0.5
        return mask_soft, mask_bin, mask_soft_vis, mask_bin_vis

    def _mask_reshape(self, mask_seq, size, patch=(1, 2, 2), scale_factor=2):
        '''
        ✨
        mask_seq: [B, Lq]
        mask_map: [B, Ttok, Htok, Wtok]
        '''
        view_size = (mask_seq.size(0), *map(lambda s, p: s // p, size, patch))
        mask_map = mask_seq.view(view_size)
        if scale_factor != 1:
            mask_map = F.interpolate(mask_map.float(), scale_factor=scale_factor) > 0.5
        return mask_map


    def _initialize_noise_statistics(
        self, reuse_noise_temporal_mean=False
    ):
        if reuse_noise_temporal_mean:
            self.noise_temporal_mean = dict()
            self.noise_temporal_mean_fg = dict()
            self.noise_temporal_mean_bg = dict()
        else:
            self.noise_temporal_mean = None
            self.noise_temporal_mean_fg = None
            self.noise_temporal_mean_bg = None

    def _reuse_noise_statistics(
        self, noise: torch.Tensor, step_idx: int, 
        ema_factor: float = 0.5, fg_mask=None,
        alpha_prog=2, alpha_mixed=1,
    ):
        if self.noise_temporal_mean is not None:
            if step_idx not in self.noise_temporal_mean.keys():
                self.noise_temporal_mean[step_idx] = noise
            else:
                noise = self.noise_temporal_mean[step_idx].flip(1) * alpha_prog / (1 + alpha_prog ** 2) ** 0.5 + \
                    noise * 1 / (1 + alpha_prog ** 2) ** 0.5
                self.noise_temporal_mean[step_idx] = noise
        
        return noise