import os
import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import mediapipe as mp
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append(os.path.realpath("../../../../src"))
from common.utils.preparation.align_mouth import landmarks_interpolate

class LandmarkDataset3D(Dataset): # video에서 audio만 추출
    def __init__(self, video_path, transform=None):
        self.video_path = glob(os.path.join(video_path, "*", "*.mp4"))
        self.video_path = [path for path in self.video_path if not "crop" in path]
        self.transform = transform
        
    def __getitem__(self, idx):
        filename = self.video_path[idx]
        frames, audio, metadata = torchvision.io.read_video(filename, output_format='THWC', pts_unit='sec')
        
        if self.transform:
            frames = self.transform
        
        lm_path = filename[:-4]+".pkl"
        landmarks_2D = pd.read_pickle(lm_path)
        preprocessed_landmarks = landmarks_interpolate(landmarks_2D)
        return frames.permute(0, 3, 1, 2), np.array(preprocessed_landmarks), filename.split("/")[-2], os.path.basename(filename)[:-4], lm_path, metadata['video_fps']
    
    def __len__(self):
        return len(self.video_path)

if __name__ == "__main__":
    dataset = LandmarkDataset3D("/app/lrs3/one_video_test")
    print(dataset.__len__())
    
    frames, preprocessed_landmarks, dir_path, filename, meta = dataset[0]
    print(frames.shape)
    print(preprocessed_landmarks[0].shape)
