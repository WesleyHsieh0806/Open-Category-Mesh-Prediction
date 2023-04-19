import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from omegaconf import DictConfig, OmegaConf
import hydra

from models import MeshRCNN
@hydra.main(version_base=None, config_path="./configs", config_name="baseline")
def main(cfg: DictConfig):
    model = MeshRCNN(cfg).cuda(2)
    print(model)

    image = torch.zeros([2, 3, 224, 224]).cuda(2)
    print(model(image))

if __name__ == "__main__":
    main()