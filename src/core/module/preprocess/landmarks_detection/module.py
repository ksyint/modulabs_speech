import torch
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
                 detection: torch.nn.Module = None,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './landmark')-> None:
        super().__init__()
        self.model = backbone
        self.dataloader = DataLoader(dataset["preprocess"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.detection = detection
        self.save_path = save_path
        
        self.bboxes_dic = {}
        self.bbox_save_path = save_path+'/bboxes.json'
        
        if os.path.exists(self.bbox_save_path):
            with open(self.bbox_save_path, "r") as jj:
                self.bboxes_dic = json.load(jj)
            print('bbox is loaded')
    
    def test_dataloader(self) -> DataLoader:
        return self.dataloader

    def step(self, frames:torch.Tensor, dirname, filename) -> dict:
        
        bboxes = [{"xmin":0, "ymin":0, "w":255, "h":255} for _ in range(frames.shape[1])]
        
        if self.detection:
            if dirname[0] in self.bboxes_dic:
                if filename[0] in self.bboxes_dic[dirname[0]]:
                    bboxes = self.bboxes_dic[dirname[0]][filename[0]]
        
            else:
                bboxes = self.detection(frames)
                
                if bboxes is None:
                    bboxes = [{"xmin":0, "ymin":0, "w":255, "h":255} for _ in range(frames.shape[1])]
                    
                    
                if not dirname[0] in self.bboxes_dic:
                    self.bboxes_dic[dirname[0]] = {}
                self.bboxes_dic[dirname[0]][filename[0]] = bboxes
        else:
            bboxes = [{"xmin":0, "ymin":0, "w":255, "h":255} for _ in range(frames.shape[1])]
        print(dirname, filename)
        results = self.model(frames, bboxes)

        return results

    def predict_step(self, batch, batch_idx) -> None:
        video, dirname, filename, info = batch
        folderpath = make_dir(self.save_path, dirname[0])
        json_save_path = os.path.join(folderpath, filename[0]+".json")
        results = self.step(video, dirname, filename)
        with open(json_save_path, "w") as json_file:
            json.dump(results, json_file, indent='\t')
            
        with open(self.bbox_save_path, 'w') as jj:
            json.dump(self.bboxes_dic, jj, indent=4)
        