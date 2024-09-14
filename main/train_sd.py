import matplotlib
matplotlib.use('Agg')
from main.utils import prepare_images_for_saving, draw_valued_array, draw_probability_histogram
from main.test_folder_sd import sample
from yt_tools.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot, copy_logs_to_logs_path
from main.sd_image_dataset import SDImageDatasetLMDB
from transformers import CLIPTokenizer, AutoTokenizer
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.utils import SDTextDataset, cycle 
from main.sd_unified_model import SDUniModel
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)
from pathlib import Path
import argparse 
import shutil 
import wandb
import logging
import torch 
import time 
import os

logger = get_logger(__name__)

# Trainer class
# ----------------------------------------------------------------------------------------------------------------------
class Trainer:

    # Init
    # ------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        self.args = args

        # Utils
        ########################################################
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        log_path = Path(args.output_path, args.log_path)
        accelerator_project_config = ProjectConfiguration(logging_dir=log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="tensorboard",
            project_config=accelerator_project_config,
            kwargs_handlers=None,
            dispatch_batches=False
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        set_seed(args.seed + accelerator.process_index)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(self.cache_dir, exist_ok=False)
            self.output_path = output_path
            os.makedirs(log_path, exist_ok=True)
        ########################################################

        # Models and datasets
        ########################################################
        self.model = SDUniModel(args, accelerator)
        self.max_grad_norm = args.max_grad_norm
        self.denoising = args.denoising
        self.step = 0 

        if self.denoising: assert args.sdxl, "denoising only supported for sdxl." 

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading ckpt only from {args.ckpt_only_path}")
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=False))
            print(self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator ckpt from {args.generator_ckpt_path}")
            print(self.model.feedforward_model.load_state_dict(torch.load(args.generator_ckpt_path, map_location="cpu"), strict=True))

        self.sdxl = args.sdxl 
        
        if self.sdxl:
            tokenizer_one = AutoTokenizer.from_pretrained(
                args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
            )

            tokenizer_two = AutoTokenizer.from_pretrained(
                args.model_id, subfolder="tokenizer_2", revision=args.revision, use_fast=False
            )
            self.tokenizers = [tokenizer_one, tokenizer_two]

            dataset = SDTextDataset(
                args.train_prompt_path, 
                is_sdxl=True,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two
            )
            eval_dataset = SDTextDataset(
                args.eval_prompt_path,
                is_sdxl=True,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two
            )

            # also load the training dataset images, this will be useful for GAN loss 
            real_dataset = SDImageDatasetLMDB(
                args.real_image_path, 
                is_sdxl=True,
                tokenizer_one=tokenizer_one,
                tokenizer_two=tokenizer_two
            )
        else:        
            tokenizer = CLIPTokenizer.from_pretrained(
                args.model_id, subfolder="tokenizer"
            )
            self.tokenizers = [tokenizer]

            uncond_input_ids = tokenizer(
                [""], max_length=tokenizer.model_max_length, 
                return_tensors="pt", padding="max_length", truncation=True
            ).input_ids.to(accelerator.device)

            dataset = SDTextDataset(args.train_prompt_path, tokenizer, is_sdxl=False)
            eval_dataset = SDTextDataset(args.eval_prompt_path, tokenizer, is_sdxl=False)
            self.uncond_embedding = self.model.text_encoder(uncond_input_ids)[0]

            # also load the training dataset images, this will be useful for GAN loss 
            real_dataset = dataset
            # TODO: REMOVE WHEN THE DATA WILL BE PREPARED
            #SDImageDatasetLMDB(
            #    args.real_image_path,
            #    is_sdxl=False,
            #    tokenizer_one=tokenizer
            #)
        ########################################################

        # Dataloader
        ########################################################
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        self.eval_dataloader = cycle(eval_dataloader)

        real_dataloader = torch.utils.data.DataLoader(
            real_dataset, num_workers=args.num_workers, 
            batch_size=args.batch_size, shuffle=True, 
            drop_last=True
        )
        real_dataloader = accelerator.prepare(real_dataloader)
        self.real_dataloader = cycle(real_dataloader)

        # use two dataloader 
        # as the generator and guidance model are trained at different paces 
        guidance_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        guidance_dataloader = accelerator.prepare(guidance_dataloader)
        self.guidance_dataloader = cycle(guidance_dataloader)

        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight
        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.previous_time = None 

        if self.denoising:
            denoising_dataloader = torch.utils.data.DataLoader(
                real_dataset, num_workers=args.num_workers, 
                batch_size=args.batch_size, shuffle=True, 
                drop_last=True
            )
            denoising_dataloader = accelerator.prepare(denoising_dataloader)
            self.denoising_dataloader = cycle(denoising_dataloader)
        ########################################################

        # fsdp utils
        ########################################################
        self.fsdp = args.fsdp
        if self.fsdp and (args.ckpt_only_path is None):
            # in fsdp hybrid_shard case, parameters initialized on different nodes may have different values
            # to fix this, we first save the checkpoint in the main process and reload it 
            generator_path = os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model.bin")
            guidance_path = os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model_1.bin")

            if accelerator.is_main_process:
                print(f"Saving current model to {args.output_path} to fix fsdp hybrid sharding's parameter mismatch across nodes")
                os.makedirs(os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
                torch.save(self.model.feedforward_model.state_dict(), generator_path)
                torch.save(self.model.guidance_model.state_dict(), guidance_path)

            accelerator.wait_for_everyone()
            generator_path = os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model.bin")
            guidance_path = os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model_1.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True))
            print(self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=True))

            if accelerator.is_main_process:
                print("reloading done")

        if self.fsdp:
            # the self.model is not wrapped in fsdp, only its two subnetworks are wrapped 
            self.model.feedforward_model, self.model.guidance_model = accelerator.prepare(
                self.model.feedforward_model, self.model.guidance_model
            )
        ########################################################

        # Optimizer
        ########################################################
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad], 
            lr=args.guidance_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        if self.fsdp: 
            (
                self.optimizer_generator, self.optimizer_guidance, 
                self.scheduler_generator, self.scheduler_guidance 
            ) = accelerator.prepare(
                self.optimizer_generator, self.optimizer_guidance, 
                self.scheduler_generator, self.scheduler_guidance 
            ) 
        else:
            # the self.model is not wrapped in ddp, only its two subnetworks are wrapped 
            (
                self.model.feedforward_model, self.model.guidance_model, self.optimizer_generator,
                self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance 
            ) = accelerator.prepare(
                self.model.feedforward_model, self.model.guidance_model, self.optimizer_generator, 
                self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance
            )
        ########################################################

        self.accelerator = accelerator
        self.log_path = log_path
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution 
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.latent_resolution = args.latent_resolution
        self.grid_size = args.grid_size
        self.log_loss = args.log_loss
        self.latent_channel = args.latent_channel

        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)
    # ------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------
    def fsdp_state_dict(self, model):
        fsdp_fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
        ):
            checkpoint = model.state_dict()

        return checkpoint
    # ------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------
    def load(self, checkpoint_path):
        # this is used for non-fsdp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")
    # ------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------
    def save(self):
        # NOTE: we save the checkpoints to two places 
        # 1. output_path: save the latest one, this is assumed to be a permanent storage
        # 2. cache_dir: save all checkpoints, this is assumed to be a temporary storage
        # training states 
        # If FSDP is used, we only save the model parameter as I haven't figured out how to save the optimizer state without oom yet, help is appreciated.
        # Otherwise, we use the default accelerate save_state function 
        
        if self.fsdp:
            feedforward_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
            guidance_model_state_dict = self.fsdp_state_dict(self.model.guidance_model)

        if self.accelerator.is_main_process:
            output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(output_path, exist_ok=True)
            print(f"start saving checkpoint to {output_path}")

            if self.fsdp: 
                torch.save(feedforward_state_dict, os.path.join(output_path, f"pytorch_model.bin"))
                del feedforward_state_dict
                torch.save(guidance_model_state_dict, os.path.join(output_path, f"pytorch_model_1.bin"))
                del guidance_model_state_dict
            else:
                self.accelerator.save_state(output_path) 

            # remove previous checkpoints 
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

            # copy checkpoints to cache 
            # overwrite the cache
            if os.path.exists(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"))

            shutil.copytree(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[:-self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.cache_dir, folder))
            print("done saving")
        torch.cuda.empty_cache()
    # ------------------------------------------------------------------------------------------------------

    # Training fn
    # ------------------------------------------------------------------------------------------------------
    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator
        if accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            accelerator.init_trackers('dmd2', config=tracker_config)

        # Data sampling
        ##############################################################################
        # 4 channel for SD-VAE, please adapt for other autoencoders 
        noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution, device=accelerator.device)
        visual = self.step % self.wandb_iters == 0
        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        if COMPUTE_GENERATOR_GRADIENT:
            text_embedding = next(self.dataloader)
        else:
            text_embedding = next(self.guidance_dataloader) 

        if self.sdxl:
            # SDXL uses zero as the uncond_embedding
            uncond_embedding = None 
        else:
            text_embedding = text_embedding['text_input_ids_one'].squeeze(1) # actually it is tokenized text prompts
            uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

        if self.denoising:
            denoising_dict = next(self.denoising_dataloader)
        else:
            denoising_dict = None

        if self.cls_on_clean_image:
            real_train_dict = next(self.real_dataloader) 
        else:
            real_train_dict = None
        ##############################################################################

        # Generator sampling/updating
        ##############################################################################
        generator_loss_dict, generator_log_dict = self.model(
            noise, text_embedding, uncond_embedding, 
            visual=visual, denoising_dict=denoising_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            real_train_dict=real_train_dict,
            generator_turn=True,
            guidance_turn=False
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0 

        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            if self.cls_on_clean_image and self.gen_cls_loss:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight
                
            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
            self.optimizer_generator.step()
            
            # if we also compute gan loss, the classifier may also receive gradient 
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad() 
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()
        ##############################################################################

        # Fake diffusion and classifier updating
        ##############################################################################
        guidance_loss_dict, guidance_log_dict = self.model(
            noise, text_embedding, uncond_embedding, 
            visual=visual, denoising_dict=denoising_dict,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0 

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.cls_on_clean_image:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight

        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad() # zero out the generator's gradient as well
        self.scheduler_guidance.step()
        ##############################################################################

        # Logging
        ##############################################################################
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        generated_image_mean = log_dict["guidance_data_dict"]['image'].mean()
        generated_image_std = log_dict["guidance_data_dict"]['image'].std()

        generated_image_mean = accelerator.gather(generated_image_mean).mean()
        generated_image_std = accelerator.gather(generated_image_std).mean()

        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                dmtrain_pred_real_image_mean = log_dict['dmtrain_pred_real_image'].mean()
                dmtrain_pred_real_iamge_std = log_dict['dmtrain_pred_real_image'].std()

                dmtrain_pred_real_image_mean = accelerator.gather(dmtrain_pred_real_image_mean).mean()
                dmtrain_pred_real_iamge_std = accelerator.gather(dmtrain_pred_real_iamge_std).mean()

                dmtrain_pred_fake_image_mean = log_dict['dmtrain_pred_fake_image'].mean()
                dmtrain_pred_fake_image_std = log_dict['dmtrain_pred_fake_image'].std()

                dmtrain_pred_fake_image_mean = accelerator.gather(dmtrain_pred_fake_image_mean).mean()
                dmtrain_pred_fake_image_std = accelerator.gather(dmtrain_pred_fake_image_std).mean()

        if self.denoising:
            original_image_mean = denoising_dict["images"].mean()
            original_image_std = denoising_dict["images"].std()

            original_image_mean = accelerator.gather(original_image_mean).mean()
            original_image_std = accelerator.gather(original_image_std).mean()

        if accelerator.is_main_process and self.log_loss and (not visual):
            wandb_loss_dict = {
                "loss_fake_mean": guidance_loss_dict['loss_fake_mean'].item(),
                "guidance_grad_norm": guidance_grad_norm.item(),
                "generated_image_mean": generated_image_mean.item(),
                "generated_image_std": generated_image_std.item(),
                "batch_size": len(noise)
            }

            if COMPUTE_GENERATOR_GRADIENT and (not self.args.gan_alone):
                wandb_loss_dict.update(
                    {
                        "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                        "dmtrain_pred_real_image_std": dmtrain_pred_real_iamge_std.item(),
                        "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                        "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item()
                    }
                )

            if self.denoising:
                wandb_loss_dict.update({
                    "original_image_mean": original_image_mean.item(),
                    "original_image_std": original_image_std.item()
                })

            if COMPUTE_GENERATOR_GRADIENT:
                wandb_loss_dict["generator_grad_norm"] = generator_grad_norm.item()

                if not self.args.gan_alone:
                    wandb_loss_dict["loss_dm"] = loss_dict['loss_dm'].item()
                    wandb_loss_dict["dmtrain_gradient_norm"] = log_dict['dmtrain_gradient_norm']

                if self.gen_cls_loss:
                    wandb_loss_dict.update({
                        "gen_cls_loss": loss_dict['gen_cls_loss'].item()
                    })

            if self.cls_on_clean_image:
                wandb_loss_dict.update({
                    "guidance_cls_loss": loss_dict['guidance_cls_loss'].item()
                })

            self.accelerator.log(
                wandb_loss_dict,
                step=self.step
            )
            logger.info(f"{wandb_loss_dict }")
        ##############################################################################

        # Eval and log
        ##############################################################################
        if self.step % args.checkpointing_steps == 0:
            sampled_data = sample(accelerator=accelerator,
                                  current_model=self.model.feedforward_model,
                                  vae=self.model.vae,
                                  text_encoder=self.model.text_encoder,
                                  dataloader=self.eval_dataloader,
                                  args=args)
            self.model.feedforward_model.train()
            self.model.vae = self.model.vae.to(torch.float16)

            if accelerator.is_main_process:
                visualize_images = sampled_data['all_images'][:args.test_visual_batch_size]
                _, W, H, C = visualize_images.shape
                visualize_captions = [caption.encode('utf-8') for caption in sampled_data['all_captions']][:args.test_visual_batch_size]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        for j in range(len(visualize_images)):
                            images = visualize_images[j].reshape(1, W, H, C)
                            validation_prompt = visualize_captions[j]
                            tracker.writer.add_images(validation_prompt, images, self.step, dataformats="NHWC")
                    else:
                        logger.warn(f"image logging not implemented for {tracker.name}")

            if accelerator.is_main_process:
                copy_logs_to_logs_path(self.log_path)

        self.accelerator.wait_for_everyone()
        ##############################################################################

    # ------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------
    def train(self):
        for index in range(self.step, self.train_iters):                
            self.train_one_step()
            if (not self.no_save)  and self.step % self.log_iters == 0:
                self.save()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.accelerator.log({"per iteration time": current_time-self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
    # ------------------------------------------------------------------------------------------------------

# Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_path", type=str, default="/mnt/localssd/test_stable_diffusion_coco")
    parser.add_argument("--log_path", type=str, default="log_stable_diffusion_coco")
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--train_prompt_path", type=str)
    parser.add_argument("--eval_prompt_path", type=str)
    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=150)
    parser.add_argument("--total_eval_samples", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--test_visual_batch_size", type=int, default=20)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=0)
    parser.add_argument("--sdxl", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--tiny_vae", action="store_true")
    parser.add_argument("--sd_teacher", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="apply gradient checkpointing for dfake and generator. this might be a better option than FSDP")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    parser.add_argument("--denoising", action="store_true", help="train the generator for denoising")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"

    assert not (args.fsdp and args.gradient_checkpointing), "currently, we don't support both options. open an issue for details."

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args
# ----------------------------------------------------------------------------------------------------------------------

# Running
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.train()
# ----------------------------------------------------------------------------------------------------------------------