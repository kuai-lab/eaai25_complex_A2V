import os
import sys
import math
import random
import natsort
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class AudioVisualTrainDataset(Dataset):
    def __init__(
        self, 
        data_type: str="",
        data_path: str="",
        data_csv: str="",
        data_files: list={},
        width: int=512, 
        height: int=512,
    ):
        self.data_type = data_type
        self.data_path = data_path
        self.data_files = data_files

        self.frames = []
        self.origin_frame = True
        self.encoded_frame = False

        self.resolution = width
        self.height = height
        self.center_crop = True
        self.random_flip = False
        
        self.transform = transforms.Compose(
        [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

        
    )

    def process_data(self, data_path):
        ## process audio
        # breakpoint()
        audio_name = data_path.split("/")[-1] + ".wav"
        audio_path = os.path.join(data_path , audio_name)
        # print(audio_name)
     
        # audio_name[:-4].split("_")
        ## process frames
        frames = []
        if len(data_path.split("-")) > 1 and data_path.split("/")[-2] != "horse_clip-clop" and data_path.split("/")[2] != "animal_kingdom_train" and data_path.split("/")[2] != "animal_kingdom":
            time_start = float(data_path.split("-")[-1].split("_")[0])
            time_end = float(data_path.split("-")[-1].split("_")[-1])

            during = math.ceil(round(float(time_end - time_start)) / 2) # 5 audio

            frame_len = len(os.listdir(data_path))-2
            stride = frame_len // (during)
            idx = [int(stride*v) for v in range(during+1)]
        else:
            frame_len = len(os.listdir(data_path))-2
            stride = frame_len // 5
            idx = [0,stride,stride*2,stride*3,stride*4,stride*5]

        for i in range(len(idx)-1):
            fm_half = (idx[i]+idx[i+1]) // 2            
            fm1 = (idx[i]+fm_half) // 2
            fm2 = (fm_half+idx[i+1]) // 2
            frame_name1 = data_path.split("/")[-1] + "_%04d.jpg" % fm1
            frame_name2 = data_path.split("/")[-1] + "_%04d.jpg" % fm2
            frame_path1 = os.path.join(data_path, frame_name1)
            frame_path2 = os.path.join(data_path, frame_name2)
            frame1 = Image.open(frame_path1)
            frame2 = Image.open(frame_path2)
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frames.append(frame1)
            frames.append(frame2)
        
        ## process label
        if data_path.split("/")[2] != "animal_kingdom_train" and data_path.split("/")[2] != "animal_kingdom":
            label = str(data_path.split("/")[-2])
            label = label.replace("_", " ")
        else:
            label = str(data_path.split("/")[-1])
            # print(label)

        
        # print("Audio Path :", audio_path)
        # print("Audio label :", label)

        return audio_path, frames, label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        try:
            data_path = self.data_files[index]
            
            audio_input, video_input, label = self.process_data(data_path)

            if self.origin_frame == True:
                video_input = torch.stack(video_input)
            
            if not os.path.exists(audio_input):
                return self.__getitem__((index+1) % len(self.data_files))

            return audio_input, video_input, label
            
        except Exception as e:
            # print("__GETITEM__:", e)
            # print(sys.exc_info()[0])
            # print("___Next___")
            return self.__getitem__((index+1) % len(self.data_files))


class AudioVisualTestDataset(Dataset):
    def __init__(
        self, 
        data_type: str="",
        data_path: str="",
        data_files: list={},
        width: int=512, 
        height: int=512,
    ):
        self.data_type = data_type
        self.data_path = data_path
        self.data_files = data_files

        self.frames = []
        self.origin_frame = True
        self.encoded_frame = False

        self.resolution = width
        self.height = height
        self.center_crop = True
        self.random_flip = True

        self.transform = transforms.Compose(
        [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def process_data(self, data_path):
        ## process audio
        audio_name = data_path.split("/")[-1] + ".wav"
        audio_path = os.path.join(data_path , audio_name)

        ## process frames
        frames = []
        if len(data_path.split("-")) > 1 and data_path.split("/")[-2] != "horse_clip-clop":
            time_start = float(data_path.split("-")[-1].split("_")[0])
            time_end = float(data_path.split("-")[-1].split("_")[-1])

            during = math.ceil(round(float(time_end - time_start)) / 2) # 5 audio

            frame_len = len(os.listdir(data_path))-2
            stride = frame_len // (during)
            idx = [int(stride*v) for v in range(during+1)]

            for i in range(len(idx)-1):
                fm = random.randint(idx[i], idx[i+1])
                frame_name = data_path.split("/")[-1] + "_%04d.jpg" % fm
                frame_path = os.path.join(data_path, frame_name)
                frame = Image.open(frame_path)
                frame = self.transform(frame)
                frames.append(frame)
            frames = frames[0:len(idx)-1]
        else:
            frame_len = len(os.listdir(data_path))-2
            stride = frame_len // 5
            idx = [0,stride,stride*2,stride*3,stride*4,stride*5]

            dummy_path = "/data/sieun/dataset/vggsound_curate/airplane_170740"
            idx=[0,1,2,3,4,5]

            frames = []
            for i in range(5):
                fm = random.randint(idx[i], idx[i+1])
                frame_name = dummy_path.split("/")[-1] + "_%04d.jpg" % fm
                frame_path = os.path.join(dummy_path, frame_name)
                frame = Image.open(frame_path)
                frame = self.transform(frame)
                frames.append(frame)
            frames = frames[0:5]

        ## process label
        label = str(data_path.split("/")[-2])
        label = label.replace("_", " ")

        print("Audio Path :", audio_path)
        print("Audio label :", label)

        return audio_path, frames, label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        try:
            data_path = self.data_files[index]
            audio_input, video_input, label = self.process_data(data_path)

            if self.origin_frame == True:
                video_input = torch.stack(video_input)

            if not os.path.exists(audio_input):
                return self.__getitem__((index+1) % len(self.data_files))

            return audio_input, video_input, label
            
        except Exception as e:
            print("__GETITEM__:", e)
            print(sys.exc_info()[0])
            print("___Next___")
            return self.__getitem__((index+1) % len(self.data_files))


def get_dataset(args):
    if args.train_test == "train" and args.dataname != "animal_kingdom":
        if args.data_type =="entire":
            train_files = []
            for dir in os.listdir(args.data_dir):
                class_dir = os.path.join(args.data_dir, dir)
                class_dir_list = os.listdir(class_dir)
                for cl in range(len(class_dir_list)):
                    video_dir = os.path.join(class_dir, class_dir_list[cl])
                    train_files.append(video_dir)
            
            print("TRAIN DATASET :", len(train_files))

            train_files = natsort.natsorted(train_files)

            return AudioVisualTrainDataset(
                data_type=args.data_type,
                data_path=args.data_dir,
                data_files = train_files,
                width = args.resolution,
                height = args.resolution,
                )
        else:
            train_files = []
            for dir in os.listdir(args.data_dir):
                if dir not in ["fire_crakling","splashing_water","squishing_water","thunder","volcano","waterfall_bubbling","underwater_bubbling","wind_noise"]:
                
                    class_dir = os.path.join(args.data_dir, dir)
                    class_dir_list = os.listdir(class_dir)
                    for cl in range(len(class_dir_list)):
                        video_dir = os.path.join(class_dir, class_dir_list[cl])
                        train_files.append(video_dir)
                
                    
            print("TRAIN DATASET :", len(train_files))

            train_files = natsort.natsorted(train_files)

            return AudioVisualTrainDataset(
                data_type=args.data_type,
                data_path=args.data_dir,
                data_files = train_files,
                width = args.resolution,
                height = args.resolution,
                )
                
    
    elif args.train_test == "train" and args.dataname == "animal_kingdom":
        train_files = []
        for dir in os.listdir(args.data_dir):
            video_dir = os.path.join(args.data_dir, dir)
            train_files.append(video_dir)

        print("TRAIN DATASET :", len(train_files))

        train_files = natsort.natsorted(train_files)

        return AudioVisualTrainDataset(
            data_type=args.data_type,
            data_path=args.data_dir,
            data_files = train_files,
            width = args.resolution,
            height = args.resolution,
            )

    elif args.train_test == "test":
        test_files = []
        for dir in os.listdir(args.data_dir):
            class_dir = os.path.join(args.data_dir, dir)
            class_dir_list = os.listdir(class_dir)
            for cl in range(len(class_dir_list)):
                video_dir = os.path.join(class_dir, class_dir_list[cl])
                test_files.append(video_dir)

        print("TEST DATASET :", len(test_files))

        return AudioVisualTestDataset(
            data_type=args.data_type,
            data_path=args.data_dir,
            data_files = test_files,
            width = args.resolution,
            height = args.resolution,
            )