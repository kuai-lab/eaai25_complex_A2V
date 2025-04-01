import argparse
import logging
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

import sys
sys.path.append(current_dir + "/..")

import time
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint
import torchvision
torchvision.disable_beta_transforms_warning()

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
import open_clip
from resampler import AudioResampler
from ImageBind.imagebind import data
from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType
from sound_dataset import get_dataset
from torch.utils.data import DataLoader
import torch.distributed as dist

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder):

    LAYERS = [
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text) ## All clip models use 77 as context length
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
    

check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def get_label(label_dict, filename):

    label_name = label_dict[filename]
    return label_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--tracker_project_name", type=str, default="EAAI24")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument('--train_test', type=str, default="train", help='train or test')
    parser.add_argument('--data_type', type=str, default="entire", help='audio input path')
    parser.add_argument("--data_dir", type=str, default="/mnt/sda/sieun/Maestro/sound_train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--interpolation", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--output_dir", type=str, default="/mnt/sda/sieun/Maestro/output_default")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."))
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_train_steps", type=int, default=400000, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'))
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    parser.add_argument('--num_quries', type=int, default=10, help='AudioPerceiver number of quries')
    parser.add_argument("--denoising_loss", type=float, default=0.5)
    parser.add_argument("--mse_loss", type=float, default=0.5)
    parser.add_argument("--dataname", type=str, default="sound_train")
    parser.add_argument("--label_path", type=str, default="./animal_kingdom_label.txt")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    return args

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    ## Handle the repository creation
    if accelerator.is_main_process:
        # ============= WanDB ============= #
        if is_wandb_available():
            import wandb
            wandb.init(project='EAAI24')
        
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = FrozenOpenCLIPEmbedder()
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=None,
    )

    AudioPerceiver = AudioResampler(
        dim=1024,
        depth=6,
        dim_head=64,
        heads=12,
        num_queries=args.num_quries,
        embedding_dim=768, # imagebind embedding size
        output_dim=1024,
        ff_mult=4,
        video_length=1, # using frame-wise version or not
    )
    AudioPerceiver.requires_grad_(True)
    AudioPerceiver.train()
    
    Imagebind = imagebind_model.imagebind_huge(pretrained=True)
    Imagebind.requires_grad_(False)
    Imagebind.eval()

    ## Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.train()

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if model.__class__ == UNet2DConditionModel:
                        state_dicts = {}
                        for name, param in model.named_parameters():
                            if not 'attn2' in name:
                                continue
                            if 'to_k' in name or 'to_v' in name:
                                state_dicts[name] = param
                        torch.save(state_dicts, os.path.join(output_dir, "unet_kv.pth"))

                    else:
                        torch.save(model.state_dict(), os.path.join(output_dir, "perceiver.pth"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                if model.__class__ == UNet2DConditionModel:
                    if os.path.exists(input_dir + '/unet_kv.pth'):
                        unet_checkpoint = torch.load(input_dir + '/unet_kv.pth')
                        for name, module in unet.named_modules():
                            name = name + '.weight'
                            if name in set(unet_checkpoint.keys()):
                                module.weight = unet_checkpoint[name]
                else:
                    if os.path.exists(input_dir + '/perceiver.pth'):
                        perceiver_checkpoint = torch.load(input_dir + '/perceiver.pth')
                        for name, module in AudioPerceiver.named_modules():
                            name = name + '.weight'
                            if name in set(perceiver_checkpoint.keys()):
                                module.weight = perceiver_checkpoint[name]

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    params_to_optim = []
    for name, param in AudioPerceiver.named_parameters():
        if param.requires_grad == True:
            params_to_optim.append(param)

    for name, param in unet.named_parameters():
        if not 'attn2' in name:
            continue
        if 'to_k' in name or 'to_v' in name:
            param.requires_grad = True
            params_to_optim.append(param)

    optimizer = optimizer_cls(
        params_to_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.train_batch_size,
                            shuffle=True,
                            num_workers=args.dataloader_num_workers
                            )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    ## Prepare everything with our `accelerator`.
    unet, AudioPerceiver, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, AudioPerceiver, optimizer, train_loader, lr_scheduler
    )

    ## For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    ## as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    ## Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    Imagebind.to(accelerator.device, dtype=weight_dtype)

    ## We need to initialize the trackers we use, and also store our configuration.
    ## The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    ## Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            ## Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            input_dir = os.path.join(args.output_dir, path)
            accelerator.load_state(input_dir)
            global_step = int(path.split("-")[1])

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
        position=0,
    )

    label_dict={}
    with open(args.label_path, 'r') as file:
        for line in file:
            file_name = line.strip().split()[0]
            parts = line.strip().split()[1:]
            label = ' '.join(parts)
            
            
            label_dict[file_name] = label

    while True:
        train_loss = 0.0
        for (audio, frames, label) in tqdm(train_loader, total=len(train_loader), position=accelerator.local_process_index+2):
            start_epoch = time.time()
            with accelerator.accumulate(unet):
                num_frames = frames.shape[1]
                with torch.no_grad():
                    if num_frames != 10:
                        audio_inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(audio, accelerator.device, clips_per_video=num_frames//2),}                     
                    else:
                        audio_inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(audio, accelerator.device),}
                    audio_inputs['audio'] = audio_inputs['audio'].type(weight_dtype)
                    output_5local, output_5clips = Imagebind(audio_inputs)
                
                output_5local = torch.cat([output for output in output_5local])

                if args.interpolation:
                    output_imagebind = output_5local
                    output_local = output_5local[0].unsqueeze(0)
                    for i in range(output_imagebind.shape[0]-1):
                        output_half = (output_imagebind[i]+output_imagebind[i+1]) / 2
                        output_local = torch.cat([output_local, output_half.unsqueeze(0), output_imagebind[i+1].unsqueeze(0)])
                    output_5local = output_local
                
                audio_emb = torch.cat([AudioPerceiver(output.unsqueeze(0)) for output in output_5local]) 
                text_emb_part = torch.cat([text_encoder.encode([""])[0].cuda().float()]).unsqueeze(0)
                if args.interpolation:
                    if num_frames != 10:
                        text_emb = torch.cat([text_emb_part for _ in range(num_frames-1)])
                    else:
                        text_emb = torch.cat([text_emb_part for _ in range(9)])
                else:
                    if num_frames != 10:
                        text_emb = torch.cat([text_emb_part for _ in range(num_frames)])
                    else:
                        text_emb = torch.cat([text_emb_part for _ in range(5)])

                text_emb[:,1:args.num_quries+1,:] = audio_emb
                audio_emb = text_emb 
                
                frames = torch.cat([frame for frame in frames]) # (b,5,3,512,512) -> (b*5,3,512,512)
                if args.interpolation:
                    frames = frames[0:num_frames-1,:,:,:]

                latents = vae.encode(frames.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                ## Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # noisy_latents.shape == (b*5,4,64,64)

                if args.dataname =="animal_kingdom":
                    # print(label_dict)
                    # breakpoint()
                    label = get_label(label_dict, label[0])
                    label = (label,)
                    # breakpoint()

                if args.interpolation:
                    encoder_hidden_states = torch.cat([text_encoder.encode([label_]).repeat(num_frames-1,1,1) for label_ in label]) 
                else:
                    encoder_hidden_states = torch.cat([text_encoder.encode([label_]).repeat(num_frames,1,1) for label_ in label])

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                ## Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, audio_emb, return_dict=False)[0]

                # ============= Loss update ============= #
                denoising_loss = 0
                mse_loss = 0

                if args.denoising_loss != 0:
                    denoising_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if args.mse_loss != 0:
                    mse_loss = F.mse_loss(audio_emb.float(), encoder_hidden_states.float())
                    
                loss = args.denoising_loss * denoising_loss + args.mse_loss * mse_loss                
                train_loss = accelerator.gather(loss).mean()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # ======================================= #
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                        "train_loss": train_loss,
                        "progress": progress_bar.n,
                        "epoch_time": time.time()-start_epoch
                    })

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                print("Training complete. Exiting...")
                accelerator.wait_for_everyone()
                accelerator.end_training()
                sys.exit()

if __name__ == "__main__":
    main()
