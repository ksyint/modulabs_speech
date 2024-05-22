import torch
import torchvision
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset
import lightning as pl
import os
import sys
sys.path.append(os.path.realpath("../../src/"))
from common.utils.set_dataloader import set_dataloader
import time

class UnsupModule(pl.LightningModule):
    def __init__(self,
                 model: Module, 
                 criterion: Module, 
                 optimizer: Module,
                 dataset: dict,
                 batch_size: int, num_workers: int, print_img: bool):
        
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = set_dataloader(dataset, batch_size, num_workers)
        self.validation_step_outputs = []
        self.testing_step_outputs = []
        self.print_img = print_img
        
    def train_dataloader(self):
        return self.dataloader["train"]
    
    def val_dataloader(self):
        return self.dataloader["validation"]
    
    def test_dataloader(self):
        return self.dataloader["test"]
    
    def configure_optimizers(self):
        return self.optimizer
    
    def step(self, input_values:torch.Tensor, attention_mask:torch.Tensor)->torch.Tensor:
        images = self.model(input_values, attention_mask)
        return images

    def training_step(self, batch, batch_idx: int) -> dict:
        input_values, attention_mask, target = batch
        before = time.time()
        images_ = self.step(input_values, attention_mask)

        print("model", time.time()-before)
        before = time.time()
        images_  = torch.mean(images_, dim=2).unsqueeze(2) # lip2text가 grayscale
        lengths = torch.Tensor([images_.shape[1]]).to(images_.device)
        loss, loss_ctc, loss_att, acc = self.criterion(images_, lengths, target)
        print("loss", time.time()-before)
        # if self.print_img and batch_idx % 10 == 0:
        #     print_img = images_[0, 0]
        #     grid = torchvision.utils.make_grid(print_img,nrow=1)
        #     self.logger.experiment.add_image('train_img', grid, global_step=batch_idx)     
        return {"loss":loss, "loss_ctc": loss_ctc, "loss_att": loss_att, "acc": acc}
    
    def on_training_epoch_end(self, outputs: list) -> None:
        for key, value in outputs.items():
            value_list = [output[key] for output in outputs]
            self.logger("train/{}".format(key), np.mean(value_list), on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx: int) -> dict:
        input_values, attention_mask, target = batch
        images_ = self.step(input_values, attention_mask)
        images_  = torch.mean(images_, dim=2).unsqueeze(2)
        lengths = torch.Tensor([images_.shape[1]]).to(images_.device)
        loss, loss_ctc, loss_att, acc = self.criterion(images_, lengths, target)
        results = {"loss": loss, "loss_ctc": loss_ctc, "loss_att": loss_att, "acc": acc}
        #print("Val_results:", results)
        self.validation_step_outputs.append(results)
        return results
    
    def on_validation_epoch_end(self) -> None:
        loss_list, ctc_list, att_list, acc_list = [], [], [], []
        for output in self.validation_step_outputs:
            loss_list.append(output["loss"].item())
            ctc_list.append(output["loss_ctc"].item())
            att_list.append(output["loss_att"].item())
            acc_list.append(output["acc"])
        self.validation_step_outputs.clear()
        self.log("valid/loss", np.mean(loss_list), on_epoch=True, sync_dist=True)
        self.log("valid/ctc_loss", np.mean(ctc_list), on_epoch=True, sync_dist=True)
        self.log("valid/att_loss", np.mean(att_list), on_epoch=True, sync_dist=True)
        self.log("valid/acc", np.mean(acc_list), on_epoch=True, sync_dist=True)


    # def testing_step(self, batch, batch_idx: int) -> dict:
    #     input_, target_, bone_mask_, tissue_mask_, imgName = batch
    #     output_ = self.step(input_)
    #     loss = self.criterion(output_, target_, tissue_mask_)
    #     pcc_list, ssim_list, psnr_list, mse_list = [], [], [], []
    #     for idx in range(input_.shape[0]):
    #         _pcc, _ssim, _psnr, _mse = self._metric(target_[idx].cpu().numpy(), output_[idx].cpu().numpy())
    #         pcc_list.append(_pcc)
    #         ssim_list.append(_ssim)
    #         psnr_list.append(_psnr)
    #         mse_list.append(_mse)
    #         if self.save_output_only:
    #             save_as_dicom(output=output_[idx].cpu().numpy(),
    #                           test_save_path=self.test_save_path,
    #                           imgName=imgName[idx])
    #         else:
    #             save_as_dicom(input=input_[idx].cpu().numpy(), 
    #                           target_=target_[idx].cpu().numpy(), 
    #                           output=output_[idx].cpu().numpy(), 
    #                           test_save_path=self.test_save_path, 
    #                           imgName=imgName[idx])
    #     results = {"loss":loss, "pcc": pcc_list, "ssim": ssim_list, "psnr": psnr_list, "mse": mse_list, "imgName": imgName}
    #     self.testing_step_outputs.append(results)
    #     return results
    
    # def on_testing_step_end(self) -> None:
    #     loss_list, pcc_list, ssim_list, psnr_list, mse_list = [], [], [], [], []
    #     for output in self.testing_step_outputs:
    #         loss_list.append(output["loss"].item())
    #         pcc_list.extend(output["pcc"])
    #         ssim_list.extend(output["ssim"])
    #         psnr_list.extend(output["mse"])
    #     print("------------------")
    #     print("Evaluation Result")
    #     print(f"pcc: {np.mean(pcc_list)}")
    #     print(f"ssim: {np.mean(ssim_list)}")
    #     print(f"psnr: {np.mean(psnr_list)}")
    #     print(f"psnr: {np.mean(mse_list)}")