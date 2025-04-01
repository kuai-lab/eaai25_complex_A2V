import argparse, os, sys, yaml
from pprint import pprint
import datetime, time
import numpy as np
from omegaconf import OmegaConf
import torch
# from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

sys.path.append('ImageBind')
from scripts.evaluation.audio_resampler import AudioResampler
from ImageBind.imagebind import data
from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType

import librosa

from TrailBlazer.Utils import keyframed_bbox
XFORMERS_MORE_DETAILS=1
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=20240524, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--original", type=str2bool, const=True, default=False, nargs="?")
    ## Edited part for NeurIPS 2024
    parser.add_argument("--step_ctrl", type=int, default=None, help="injection step")
    parser.add_argument("--audio_path1", type=str, default=None, help="audio input")
    parser.add_argument("--audio_path2", type=str, default=None, help="audio input")
    parser.add_argument("--audio_path3", type=str, default=None, help="audio input")
    parser.add_argument("--audio_path4", type=str, default=None, help="audio input")
    parser.add_argument("--num_queries", type=int, default=None, help="perceiver num quries")
    parser.add_argument("--audio_ckpt_path", type=str, default=None, help="perceiver")
    parser.add_argument("--verbose", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--trailblazer_config", type=str, default=None, help="Trailblazer config file path")
    parser.add_argument("--pos", type=str, default="LR")
    parser.add_argument("--application", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--long_video", type=str2bool, const=True, default=False, nargs="?")

    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    
def config_loader(filepath):
    data = None
    with open(filepath, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        yamlfile.close()
    return data

def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 0: multi-gpu setting
    ## -----------------------------------------------------------------
    # seed_list = list(map(int, args.seed.split(',')))
    seed_list = list(range(0,10000,100))

    num_samples = len(seed_list)
    samples_split = num_samples // gpu_num
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    seed_list_rank = [seed_list[i] for i in indices]


    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    noise_shape = [1, channels, frames, h, w]
    os.makedirs(args.savedir, exist_ok=True)

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    Imagebind = imagebind_model.imagebind_huge(pretrained=True)
    Imagebind.requires_grad_(False)
    Imagebind.eval()
    Imagebind.to("cuda")
    AudioPerceiver = AudioResampler(num_queries=args.num_queries, depth=6)
    AudioPerceiver.eval()
    state_dict = torch.load(args.audio_ckpt_path + '/perceiver.pth')
    # state_dict = torch.load("checkpoints/test/perceiver-6.pth")
    AudioPerceiver.load_state_dict(state_dict)
    AudioPerceiver = AudioPerceiver.to('cuda')

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in enumerate(range(0, len(seed_list_rank))):
            print(f'[rank:{gpu_no}] batch-{indice+1} ({args.bs})x{args.n_samples} ...')
            seed = seed_list_rank[indice]
            # seed_everything(seed)
            
            uc_emb = model.get_learned_conditioning([""])
            uc_emb = uc_emb.repeat(args.frames,1,1)
            uc_emb_copied = uc_emb.detach().clone()

            # audio 1
            sample_rate = 44100
            audio, sample_rate = librosa.load(args.audio_path1, sr=sample_rate, mono=True)
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            number_audio = int(duration // 2)
            print("Audio1's number_audio:", number_audio)
            audio_inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data([args.audio_path1], 'cuda', clips_per_video=number_audio),}
            audio_inputs['audio'] = audio_inputs['audio']

            output_5local, _ = Imagebind(audio_inputs)
            
            batch_size, temporal_length, num_tokens, token_dim = output_5local.shape
            output_5local = output_5local.reshape((batch_size*temporal_length, num_tokens, token_dim))
            audio_embedding = torch.cat([AudioPerceiver(output.unsqueeze(0)) for output in output_5local])
            text_emb1 = torch.cat([audio_embedding[i].unsqueeze(0).repeat_interleave(8,dim=0) for i in range(number_audio)], dim=0).to('cuda') # 1 10 1024 -> 8 10 1024 -> 40 10 1024

            # audio 2
            sample_rate = 44100
            audio, sample_rate = librosa.load(args.audio_path2, sr=sample_rate, mono=True)
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            number_audio = int(duration // 2)
            print("Audio2's number_audio:", number_audio)
            audio_inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data([args.audio_path2], 'cuda', clips_per_video=number_audio),}
            audio_inputs['audio'] = audio_inputs['audio']

            output_5local, _ = Imagebind(audio_inputs)
            
            batch_size, temporal_length, num_tokens, token_dim = output_5local.shape
            output_5local = output_5local.reshape((batch_size*temporal_length, num_tokens, token_dim))
            audio_embedding = torch.cat([AudioPerceiver(output.unsqueeze(0)) for output in output_5local])
            text_emb2 = torch.cat([audio_embedding[i].unsqueeze(0).repeat_interleave(8,dim=0) for i in range(number_audio)], dim=0).to('cuda') # 5 10 1024 -> 1 10 1024 -> 8 10 1024 -> 16 10 1024
            
            split_size = args.frames // 8
            try:
                uc_emb[:, 1:args.num_queries+1] = text_emb1[:args.frames]
            except:
                for i in range(split_size):
                    uc_emb[i*8:(i+1)*8, 1:args.num_queries+1] = text_emb1[:8]

            try:
                uc_emb[:, args.num_queries+1:2*args.num_queries+1] = text_emb2[:args.frames]
            except:
                for i in range(split_size):
                    uc_emb[i*8:(i+1)*8, args.num_queries+1:2*args.num_queries+1] = text_emb2[:8]

            if args.mode == 'base':
                cond = {"c_crossattn": [uc_emb], "fps": args.fps}

            #---trailblazer
            bundle = None
            bbox_per_frame = None
            if args.trailblazer_config is not None:
                bundle = config_loader(args.trailblazer_config)
                bbox_per_frame = keyframed_bbox(bundle, args.frames)

            context_next=None

            ## inferenced
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.verbose, args.n_samples, \
                                                    args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, args=args, 
                                                    bundle=bundle, bbox_per_frame=bbox_per_frame, uc_emb_copied=uc_emb_copied,
                                                    context_next=context_next,
                                                    **kwargs)
            save_videos(batch_samples, args.savedir, [f"seed{seed}"], fps=args.savefps)


        print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@Maestro Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    # seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)