import os
import json
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from glob import glob
from torch import nn
import sys
sys.path.append("../../../../../src")
from common.utils import Sobel, GaussianBlur, DistanceCrop
from glob import glob

class VAE_Dataset(Dataset):
    def __init__(self, json_path:str = None, dir_path:str = None, type:str = None, size:int = 64)->None:
        if json_path is None:
            self.video_path = glob(os.path.join(dir_path, "*", "*.mp4"))
        else:
            with open(json_path, "r") as json_file:
                self.video_path = json.load(json_file)[type]
                self.video_path = [os.path.join(dir_path, fname) for fname in self.video_path]
        self.flip = transforms.RandomHorizontalFlip(p=1.)
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.8),
            #transforms.RandomApply([Sobel()], p=0.6),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        filename = self.video_path[idx]
        frames, audio, metadata = read_video(filename, output_format='THWC', pts_unit='sec')
        
        img = frames
        images = [self.data_transforms(frames[i].permute((2, 0, 1))).unsqueeze(0) for i in range(frames.shape[0])]
        return torch.concat(images, dim=0)

    def __collate_fn__(self, samples):
        return torch.concat(samples, dim=0)

if __name__ == "__main__":
    dataset = VAE_Dataset(dir_path = "/app/lrs3/test", size=64)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.__collate_fn__)
    print(dataset[0].shape)
    print(hasattr(dataset, "__collate_fn__"))
    # for batch in dataloader:
    #     print(batch.shape)
    dataset = VAE_Dataset(json_path = "/app/Talking_Head_Generation_2/preprocess_log/Lrs3_train_valid.json", dir_path = "/app/lrs3/trainval", type="Validation", size=64)
    print(dataset[0].shape)