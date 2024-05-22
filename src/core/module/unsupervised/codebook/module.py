import torch
from torch.utils.data import DataLoader
import lightning as pl
import torchvision
import os
import sys
sys.path.append(os.path.realpath("../../src/"))
from common.utils.set_dataloader import set_dataloader

class VAE(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.nn.Module,
                 scheduler: torch.nn.Module,
                 dataset: dict,
                 batch_size: int, num_workers: int, save_path: str=None, print_img: bool=True):

        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = set_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        print(self.dataloader)
        self.print_img = print_img

    def train_dataloader(self):
        return self.dataloader["train"]

    def val_dataloader(self):#validation_dataloader(self):
        return self.dataloader["validation"]

    def test_dataloader(self):
        return self.dataloader["test"]
    
    def configure_optimizers(self):
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def step(self, input_:dict) -> torch.Tensor:
        output, recon_loss, contrastive_loss = self.model(input_)
        return output, recon_loss, contrastive_loss

    def training_step(self, batch, batch_idx: int) -> dict:
        input_ = batch
        output, recon_loss, contrastive_loss = self.step(input_)
        vae_recon_loss = self.criterion(output, input_)
        loss = vae_recon_loss + recon_loss + contrastive_loss

        if self.print_img and batch_idx % 10 == 0:
            print_img = torch.cat([input_[:2], output[:2]], dim=3)
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            self.logger.experiment.add_image('train_img', grid, global_step=batch_idx)

        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        self.log("train/vae_recon_loss", vae_recon_loss, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("train/contrastive_loss", contrastive_loss, on_epoch=True, sync_dist=True)
        return {"loss": loss, "vae_recon_loss": vae_recon_loss, "recon_loss": recon_loss, "contrastive_loss": contrastive_loss}
    
    # def on_training_epoch_end(self, outputs: list) -> None:
    #     for key in outputs[0].keys():
    #         loss = [output[key] for output in outputs]
    #         self.log("train/{}".format(key), np.mean(loss), on_epoch=True, sync_dist=True)
            
    def validation_step(self, batch, batch_idx: int) -> dict:
        input_ = batch
        output, recon_loss, contrastive_loss = self.step(input_)
        vae_recon_loss = self.criterion(output, input_)
        loss = vae_recon_loss + recon_loss + contrastive_loss
        self.log("valid/loss", loss, on_epoch=True, sync_dist=True)
        self.log("valid/vae_recon_loss", vae_recon_loss, on_epoch=True, sync_dist=True)
        self.log("valid/recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("valid/contrastive_loss", contrastive_loss, on_epoch=True, sync_dist=True)
        results = {"loss": loss, "vae_recon_loss": vae_recon_loss, "recon_loss": recon_loss, "contrastive_loss": contrastive_loss}
        #self.validation_step_outputs.append(results)
        return results
    
    # def on_validation_epoch_end(self) -> None:
    #     loss_list, vae_list, recon_list, contrastive_list = [], [], [], []
    #     for output in self.validation_step_outputs:
    #         loss_list.append(output["loss"].item())
    #         vae_list.append(output["vae_recon_loss"].item())
    #         recon_list.append(output["recon_loss"].item())
    #         contrastive_list.append(output["contrastive_loss"])
    #     self.validation_step_outputs.clear()
    #     self.log("valid/loss", np.mean(loss_list), on_epoch=True, sync_dist=True)
    #     self.log("valid/vae_recon_loss", np.mean(vae_list), on_epoch=True, sync_dist=True)
    #     self.log("valid/recon_loss", np.mean(recon_list), on_epoch=True, sync_dist=True)
    #     self.log("valid/contrastive_loss", np.mean(contrastive_list), on_epoch=True, sync_dist=True)


    def testing_step(self, batch, batch_idx: int) -> dict:
        input_ = batch
        output, recon_loss, contrastive_loss = self.step(input_)
        vae_recon_loss = self.criterion(output, input_)
        loss = vae_recon_loss + recon_loss + contrastive_loss
        results = {"loss": loss, "vae_recon_loss": vae_recon_loss, "recon_loss": recon_loss, "contrastive_loss": contrastive_loss}
        self.testing_step_outputs.append(results)
        return results
    
    def on_testing_step_end(self) -> None:
        loss_list, vae_list, recon_list, contrastive_list = [], [], [], []
        for output in self.validation_step_outputs:
            loss_list.append(output["loss"].item())
            vae_list.append(output["vae_recon_loss"].item())
            recon_list.append(output["recon_loss"].item())
            contrastive_list.append(output["contrastive_loss"])
        print("------------------")
        print("Evaluation Result")
        print(f"loss: {np.mean(loss_list)}")
        print(f"VAE Recon loss: {np.mean(vae_list)}")
        print(f"Recon loss: {np.mean(recon_list)}")
        print(f"Contrastive loss: {np.mean(contrastive_list)}")