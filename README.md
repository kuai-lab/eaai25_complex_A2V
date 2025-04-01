# eaai25_complex_A2V
Semantically Complex Audio to Video Generation with Audio Source Separation

## Getting Started
### Installation
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
Place downloaded weights under "./checkpoints/cim" folder.

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
If you want to use a custom dataset, only videos shorter than 10 seconds are allowed, and they should be prepared separately as frames and audio.
```plaintext
dataset/
├── video_001/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   ├── frame_N.jpg
│   ├── audio_001.wav
├── video_002/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   ├── frame_N.jpg
│   ├── audio_002.wav
└── ...
```
Specify the dataset folder path for ```--data_dir```

## Inference
```bash
$ bash scripts/run.sh
```
The ```--pos``` option represents the position of the bounding box, and you should choose between ```LR (Left & Right) or TD (Top & Down)```.

