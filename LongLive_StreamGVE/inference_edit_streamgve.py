import argparse
import torch
import os
from pathlib import Path 

import json
from collections import OrderedDict
from omegaconf import OmegaConf
import peft
import numpy as np
from PIL import Image
from einops import rearrange
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import write_video

from pipeline import (
    EditCausalInferencePipeline
)
from utils.misc import set_seed
from utils.lora_utils import configure_lora_for_model
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

from diffusers.utils import load_video


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)

def find_closest_num_frame(x, a=4, b=3):
    max_m = (x + a - 1) // (a * b)
    while max_m > 0:
        y = a * b * max_m - a + 1
        if y <= x:
            return y
        max_m -= 1

def load_pipe(args):
    config = OmegaConf.load(args.config_path)
    config['model_kwargs']['timestep_shift'] = args.flow_shift
    config['denoising_step_list'] = np.arange(1000, 0, -1000 / args.step).astype(int).tolist()
    # infinity relative rope
    config['model_kwargs']['use_infinite_attention'] = getattr(args, 'use_infinite_attention', False)

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout,
            )
        set_seed(config.seed + local_rank)
        config.distributed = True  # Mark as distributed for pipeline
        if rank == 0:
            print(f"[Rank {rank}] Initialized distributed processing on device {device}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
        config.distributed = False  # Mark as non-distributed
        print(f"Single GPU mode on device {device}")

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40

    torch.set_grad_enabled(False)

    # Initialize pipeline
    # Note: checkpoint loading is now handled inside the pipeline __init__ method
    pipeline = EditCausalInferencePipeline(config, device=device)

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
        if config.use_ema:
            def _clean_key(name: str) -> str:
                """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
                name = name.replace("_fsdp_wrapped_module.", "")
                return name

            cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
            missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

    # --------------------------- LoRA support (optional) ---------------------------

    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None) and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"LoRA enabled with config: {config.adapter}")
            print("Applying LoRA to generator (inference)...")
        
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )

        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            if local_rank == 0:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
            if local_rank == 0:
                print("LoRA weights loaded for generator")
        else:
            if local_rank == 0:
                print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

        pipeline.is_lora_enabled = True

    return pipeline, low_memory, device, local_rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--src_prompt", type=str, required=True)
    parser.add_argument("--trg_prompt", type=str, required=True)
    parser.add_argument("--src_word", type=str, required=True)
    parser.add_argument("--trg_word", type=str, required=True)
    
    # first frame condition, triple_first_frame=True for LongLive
    parser.add_argument("--first_frame_edit", type=str, default=None)
    parser.add_argument("--triple_first_frame", action="store_true", default=True)

    # hyper-parameters
    parser.add_argument("--fg_boost_factor", type=float, default=2.0, help='CrossAttn Boosting')
    parser.add_argument("--blend_power", type=float, default=2.0, help='rho')

    # model settings
    parser.add_argument("--step", type=int, default=15, help='1~1000')
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--use_infinite_attention", action="store_true", default=False)

    parser.add_argument("--config_path", type=str, default='configs/longlive_inference.yaml')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    pipeline, low_memory, device, local_rank = load_pipe(args)
    # Move pipeline to appropriate dtype and device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(Path(args.save_path).parent, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # load video
    src_video = load_video(args.data_path)
    if args.first_frame_edit is not None:
        src_first_frame = src_video[0]
        trg_first_frame = Image.open(args.first_frame_edit).convert('RGB')
    else:
        src_first_frame = None
        trg_first_frame = None

    height = src_video[0].size[1]
    width = src_video[0].size[0]
    num_frames = len(src_video)
    new_len = find_closest_num_frame(num_frames)
    src_video = src_video[: new_len]
    num_frames = len(src_video)
    print(num_frames, height, width)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # AE
    src_video_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
    video_latents = pipeline.vae.encode_to_latent(
        src_video_tensor.to(device=device, dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)

    # first frame condition
    independent_first_frame = False
    triple_first_frame = False
    if args.first_frame_edit is not None:
        independent_first_frame = True
        triple_first_frame = False
        src_first_frame = pipeline.vae.encode_to_latent(
            transform(src_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        trg_first_frame = pipeline.vae.encode_to_latent(
            transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        if args.triple_first_frame:
            independent_first_frame = False
            triple_first_frame = True
            src_first_frame = src_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]
            trg_first_frame = trg_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    edit_video = pipeline.inference(
        src_video=video_latents,
        src_prompts=args.src_prompt,
        trg_prompts=args.trg_prompt,
        src_trigger_words=args.src_word,
        trg_trigger_words=args.trg_word,
        return_latents=False,
        wo_video_decode=False,
        profile=False,
        low_memory=low_memory,

        independent_first_frame=independent_first_frame,
        triple_first_frame=triple_first_frame,
        src_initial_latent=src_first_frame,
        trg_initial_latent=trg_first_frame,

        fg_boost_factor=args.fg_boost_factor,
        blend_power=args.blend_power,
    )

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    write_video(
        args.save_path, 
        rearrange(edit_video[0], 't c h w -> t h w c').cpu() * 255, 
        fps=16
    )


