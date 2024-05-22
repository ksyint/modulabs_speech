import torch
import pickle
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
import json
import sys
import numpy as np
from natsort import natsorted
sys.path.append(os.path.realpath("./src"))
from common.utils import make_dir

class LandmarksDetection(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module,
                 dataset: dict,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './landmark')-> None:
        super().__init__()
        self.model = backbone
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.bboxes_dic = {}
        self.bbox_save_path = save_path+'/bboxes.json'
        
        if os.path.exists(self.bbox_save_path):
            with open(self.bbox_save_path, "r") as jj:
                self.bboxes_dic = json.load(jj)
            print('bbox is loaded')
    
    def test_dataloader(self) -> DataLoader:
        return self.dataloader

    def step(self, frames:torch.Tensor, landmarks) -> dict:
        rotated_landmarks, bbox_list = self.model(frames, landmarks)

        return rotated_landmarks, bbox_list

    def predict_step(self, batch, batch_idx) -> None:
        video, preprocessed_landmarks, dirname, filename, lm_path, info = batch
        print(dirname, filename)
        rotated_landmarks, bbox_list = self.step(video, preprocessed_landmarks)
        with open(lm_path[0][:-4]+"_rotated.pkl", "wb") as pickle_file:
            pickle.dump(np.array(rotated_landmarks), pickle_file)
        
        self.bboxes_dic[dirname[0]+"/"+filename[0]] = bbox_list
        with open(self.bbox_save_path, 'w') as json_file:
            json.dump(self.bboxes_dic, json_file, indent=4)
        