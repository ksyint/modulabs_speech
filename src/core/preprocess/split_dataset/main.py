import hydra
from omegaconf import DictConfig, OmegaConf
from glob import glob
from sklearn.model_selection import train_test_split
import sys
import os
import json
sys.path.append(os.path.realpath("./src"))
import core
from common.utils import seed_everything, instantiate_dict

@hydra.main(version_base=None, config_path="../../../config/preprocess", config_name="config_split_dataset")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    cfg = instantiate_dict(cfg) 
    seed_everything(cfg["seed"])
    data_list = glob(cfg["data_path"])
    data_list = [os.path.join(path.split("/")[-2], os.path.basename(path)) for path in data_list]
    if cfg["Need_Validation_only"]:
        train, validation = train_test_split(data_list, test_size=cfg["validation_size"], random_state=cfg["seed"], shuffle=cfg["shuffle"])
        data = {
            "Train": train,
            "Validation": validation
        }
    else:
        train, test = train_test_split(data_list, train_size=cfg["train_size"], random_state=cfg["seed"], shuffle=cfg["shuffle"])
        train, validation = train_test_split(train, test_size=cfg["validation_size"], random_state=cfg["seed"], shuffle=cfg["shuffle"])
        data = {
            "Train": train,
            "Validation": validation,
            "Test": test
        }

    with open(os.path.join(cfg["save_path"], cfg["filename"]), "w") as json_file:
        json.dump(data, json_file, indent='\t')

if __name__ == "__main__":
    main()