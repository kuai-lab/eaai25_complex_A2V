# Semantically Complex Audio to Video Generation with Audio Source Separation
![figure2](https://github.com/user-attachments/assets/bdd808ef-4035-400a-8909-1670db63f99d)

- Abstract: Recent advancements in artificial intelligence for audio-to-video generation have shown the ability to generate high-quality videos from audio, particularly by focusing on temporal semantics and magnitude. However, existing works struggle to capture all semantics from audio, as real world audios often consist of mixed sources, making it challenging to generate semantically aligned videos. To solve this problem, we present a novel multi-source audio-to-video generation framework that incorporates decomposed multiple audio sources into video generative models. Specifically, our proposed Attention Mosaic directly maps each decomposed audio feature to the corresponding spatial attention feature. In addition, our condition injection module is helpful for producing more natural contexts with non-audible objects by leveraging the knowledge of existing generative models. Our experiments show that the proposed framework achieves state-of-the-art performance in representing both multi- and single-source audio-to-video generation methods.

## Getting Started
### Installation
Our code is tested on Ubuntu 20.04 and cuda 11.8
- Follow the steps below:
```bash
$ conda create --name Maestro python==3.10.0
$ conda activate Maestro
$ pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
$ pip install pyyaml omegaconf pytorch_lightning discord opencv-python einops timm
$ pip install open-clip-torch==2.24.0
$ git clone https://github.com/facebookresearch/ImageBind.git
```

Clone the ImageBind repository, then replace the original ```imagebind_model.py``` and ```data.py``` with ```./change/imagebind_model.py``` and ```./change/data.py```, respectively.

### Download Pretrained Model
1. Download Link : [Condition Injection Module weights](https://drive.google.com/file/d/189-AZzkyNbqoprN44lwgNvO7wbtM0z9Z/view?usp=sharing)
```bash
$ mkdir checkpoints/cim
```
Place downloaded weights under "./checkpoints/cim" folder. (trained on VGGSound & Landscape dataset)

2. Download Link(Video diffusion weights) : https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt

 ```bash
$ cd checkpoints
$ mkdir base_512_v2
```
Place downloaded weights under "./checkpoints/base_512_v2" folder.

3. Download Link(ImageBind weights) : https://github.com/facebookresearch/ImageBind?tab=readme-ov-file

Place downloaded weights under "./checkpoints" folder.


## Training Condition Injection Module
```bash
$ bash train.sh
```

### Dataset Download
- VGGSound : https://github.com/hche11/VGGSound
- Landscape : https://kuai-lab.github.io/eccv2022sound/

Preprocess the downloaded dataset as follows:
```plaintext
PROJECT_ROOT/dataset/
├── video_001/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
│   ├── 0000N.jpg
│   ├── video_001.wav
├── video_002/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
│   ├── 0000N.jpg
│   ├── video_002.wav
└── ...
```
Specify the dataset folder path(PROJECT_ROOT/dataset) for ```--data_dir```

If you want to use custom datasets, only videos shorter than 10 seconds are allowed, and they should be prepared separately as frames and audio.

## Inference
```bash
$ bash scripts/run.sh
```
The ```--pos``` option represents the position of the bounding box, and you should choose between ```"LR" (Left & Right) or "TD" (Top & Down)```.

## Acknowlegement
Our code is based on several interesting and helpful projects:
- VideoCrafter : https://github.com/AILab-CVC/VideoCrafter
- ImageBindhttps : https://github.com/facebookresearch/ImageBind
- TrailBlazer : https://github.com/hohonu-vicml/Trailblazer
- Perceiver : https://github.com/lucidrains/perceiver-pytorch
