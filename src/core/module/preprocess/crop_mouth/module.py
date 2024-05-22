import torch
import cv2
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
import json
import sys
import dlib
import skvideo
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from tqdm import tqdm
sys.path.append(os.path.realpath("./src"))
from common.utils import make_dir
from common.utils.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from torchvision.io import read_video, write_video

class CropMouth():
    def __init__(self,
                video_dir: str, 
                face_predictor_path: str = "/app/Talking_Head_Generation/src/common/utils/preparation/shape_predictor_68_face_landmarks.dat",
                mean_face_path: str = "/app/Talking_Head_Generation/src/common/utils/preparation/20words_mean_face.npy",
                size: int = 224,
                save_path: str = './landmark')-> None:
        super().__init__()
        self.video_list = glob(os.path.join(video_dir, "*", "*.mp4"))
        # if face_predictor_path is not None:
        #     self.detector = dlib.get_frontal_face_detector()
        #     self.predictor = dlib.shape_predictor(face_predictor_path)
        # else:
        self.detector, self.predictor = None, None

        self.STD_SIZE = (size, size)
        self.mean_face_landmarks = np.load(mean_face_path)
        self.stablePntsIDs = [33, 36, 39, 42, 45]

    def detect_landmarks(self, video_path):
        videogen = skvideo.io.vread(video_path)
        frames = np.array([frame for frame in videogen])
        landmarks = []
        for frame in tqdm(frames):
            landmark = detect_landmark(frame, self.detector, self.predictor)
            landmarks.append(landmark)
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        return preprocessed_landmarks

    def predict_step(self) -> None:
        for video_path in tqdm(self.video_list):
            if "_crop" in os.path.basename(video_path):
                continue
            if self.detector is not None:
                preprocessed_landmarks = self.detect_landmarks(video_path)
            else:
                lm_path = video_path[:-4]+".pkl"
                landmarks = pd.read_pickle(lm_path)
                preprocessed_landmarks = landmarks_interpolate(landmarks)
            frames, audio, info = read_video(video_path, output_format='THWC', pts_unit='sec')
            rois, cropped_landmarks = crop_patch(video_path, preprocessed_landmarks, self.mean_face_landmarks, self.stablePntsIDs, self.STD_SIZE, 
                            window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
            
            new_frames = []
            for idx in range(rois.shape[0]):
                frame = np.zeros(rois[idx].shape)
                points = cropped_landmarks[idx]
                for point in points:
                    frame = cv2.circle(frame, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=-1) #save the original image with landmark on it
                new_frames.append(frame)
            write_video(video_path[:-4]+"_crop.mp4", 
                video_array=new_frames, #rois, 
                fps=info["video_fps"],
                audio_array=audio,
                audio_fps=info["audio_fps"],
                audio_codec="aac")