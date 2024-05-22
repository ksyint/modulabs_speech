import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
import random
import numpy as np
import os
sys.path.append(os.path.realpath("./src"))
import core
from common.utils import seed_everything, instantiate_dict
seed_everything()

@hydra.main(config_path="../../../config/preprocess", config_name="config_mouthcrop")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    cfg = instantiate_dict(cfg) 
    
    cfg["Trainer"].predict_step()
    print('all things done! It worked!')

if __name__ == "__main__":
    main()