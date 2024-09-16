from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler
from main.coco_eval.coco_evaluator import evaluate_model
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate.utils import ProjectConfiguration
from main.utils import create_image_grid
from accelerate.logging import get_logger
from main.utils import SDTextDataset
from accelerate.utils import set_seed
from accelerate import Accelerator
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import argparse
import torch.distributed as dist
import pandas as pd
import logging
import wandb
import torch
import glob
import time
import os

logger = get_logger(__name__, log_level="INFO")


def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        generator = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        ).float()
        generator.requires_grad_(False)
    else:
        generator = base_model

    # sometime the state_dict is not fully saved yet 
    counter = 0
    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            break
        except:
            print(f"fail to load checkpoint {checkpoint_path}")
            time.sleep(1)

            counter += 1

            if counter > 100:
                return None

    # # unwrap the generator 
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith("feedforward_model."):
    #         new_state_dict[k[len("feedforward_model."):]] = v

    # print(generator.load_state_dict(new_state_dict, strict=True))
    print(generator.load_state_dict(state_dict, strict=True))
    return generator


def get_x0_from_noise(sample, model_output, timestep):
    # alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    # 0.0047 corresponds to the alphas_cumprod of the last timestep (999)
    alpha_prod_t = (torch.ones_like(timestep).float() * 0.0047).reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


@torch.no_grad()
def log_validation(accelerator, tokenizer, vae, text_encoder, current_model, step, args, teacher_pipeline=None):
    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        'A sad puppy with large eyes',
        'A girl with pale blue hair and a cami tank top',
        'cute girl, Kyoto animation, 4k, high resolution',
        "A person laying on a surfboard holding his dog",
        "Green commercial building with refrigerator and refrigeration units outside",
        "An airplane with two propellor engines flying in the sky",
        "Four cows in a pen on a sunny day",
        "Three dogs sleeping together on an unmade bed",
        "a deer with bird feathers, highly detailed, full body"
    ]

    image_logs = []
    generator = torch.Generator().manual_seed(args.seed)

    for _, prompt in enumerate(validation_prompts):
        text_input_ids_one = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_one = text_input_ids_one.to(accelerator.device).reshape(-1, text_input_ids_one.shape[-1])
        text_embedding = text_encoder(text_input_ids_one)[0]

        images = []
        for _ in range(4):
            noise = torch.randn(len(text_embedding), 4,
                                args.latent_resolution, args.latent_resolution,
                                dtype=torch.float32,
                                generator=generator,
                                ).to(accelerator.device)
            timesteps = torch.ones(len(text_embedding), device=accelerator.device, dtype=torch.long)

            if teacher_pipeline:
                eval_images = teacher_pipeline(
                    prompt_embeds=text_embedding,
                    latents=noise,
                    guidance_scale=args.real_guidance_scale,
                    output_type="np",
                    num_inference_steps=50
                ).images
                eval_images = (torch.tensor(eval_images, dtype=torch.float32) * 255.0).to(torch.uint8)
            else:
                eval_images = current_model(
                    noise, timesteps.long() * (args.num_train_timesteps - 1), text_embedding
                ).sample

                eval_images = get_x0_from_noise(
                    noise, eval_images, timesteps
                )

                # decode the latents and cast to uint8 RGB
                vae = vae.to(eval_images.dtype)
                eval_images = vae.decode(eval_images * 1 / 0.18215).sample.float()
                eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                eval_images = eval_images.contiguous()
            images.append(eval_images.cpu().numpy())

        images = np.concatenate(images, axis=0)
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")


def prepare_val_prompts(path, bs=20, max_cnt=5000):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


@torch.no_grad()
def sample(accelerator, current_model, vae, tokenizer, text_encoder, prompts_path, args, teacher_pipeline=None):

    # Preparation
    ##########################################
    current_model.eval()
    set_seed(args.seed + accelerator.process_index)
    generator = torch.Generator().manual_seed(args.seed + accelerator.process_index)
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(
        prompts_path, bs=args.batch_size, max_cnt=args.total_eval_samples
    )
    ##########################################

    local_images = []
    local_text_idxs = []
    for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):

        # Inputs
        ##########################################
        text_input_ids_one = tokenizer(
            list(mini_batch),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_one = text_input_ids_one.to(accelerator.device).reshape(-1, text_input_ids_one.shape[-1])
        text_embedding = text_encoder(text_input_ids_one)[0]

        timesteps = torch.ones(len(text_embedding), device=accelerator.device, dtype=torch.long)

        noise = torch.randn(len(text_embedding), 4,
                            args.latent_resolution, args.latent_resolution,
                            dtype=torch.float32,
                            generator=generator,
                            ).to(accelerator.device)
        ##########################################

        # Sampling
        ##########################################
        if teacher_pipeline is not None:
            eval_images = teacher_pipeline(
                prompt_embeds=text_embedding,
                latents=noise,
                guidance_scale=args.real_guidance_scale,
                output_type="np",
                num_inference_steps=50
            ).images
            eval_images = (torch.tensor(eval_images, dtype=torch.float32) * 255.0).to(torch.uint8)
        else:
            # generate images and convert between noise and data prediction if needed
            eval_images = current_model(
                noise, timesteps.long() * (args.num_train_timesteps - 1), text_embedding
            ).sample

            eval_images = get_x0_from_noise(
                noise, eval_images, timesteps
            )

            # decode the latents and cast to uint8 RGB
            vae = vae.to(eval_images.dtype)
            eval_images = vae.decode(eval_images * 1 / 0.18215).sample.float()
            eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            eval_images = eval_images.contiguous().cpu()
        ##########################################

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(eval_images[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    # Aggregation
    ##########################################
    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()

    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs)

    images, prompts = [], []
    if dist.get_rank() == 0:
        gathered_images = np.concatenate(
            [images.cpu().numpy() for images in gathered_images], axis=0
        )
        gathered_text_idxs = np.concatenate(
            [text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0
        )
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])
    ##########################################

    # Done.
    dist.barrier()
    return images, prompts
