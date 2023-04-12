# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from omegaconf import DictConfig, OmegaConf
import hydra


from .backbone import get_backbone
from .roi_head import get_roi_head
# from detectron2.config import configurable
# from detectron2.data.detection_utils import convert_image_to_rgb
# from detectron2.layers import move_device_like
# from detectron2.structures import ImageList, Instances
# from detectron2.utils.events import get_event_storage
# from detectron2.utils.logger import log_first_n

# from .backbone import Backbone, build_backbone
# from ..postprocessing import detector_postprocess
# from ..proposal_generator import build_proposal_generator
# from ..roi_heads import build_roi_heads
# from .build import META_ARCH_REGISTRY

__all__ = ["MeshRCNN"]


class MeshRCNN(nn.Module):
    """
    Mesh R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation (We don't neew this one)
    3. Per-region feature extraction and prediction
    """

    def __init__(
        self,
        cfg
    ):
        """
        MeshRCNN consists of two main components
        1. backbone: used to extract image features
        2. roi-head: used to predict 3D Voxel and mesh by sending image features into it
        Args:
            cfg should contain following configurations
                backbone:
                    type: resnet50
                    pretrained: true
        """
        super().__init__()
        self.backbone = get_backbone(cfg.backbone.type, cfg.backbone.pretrained)
        self.bilinear_interpolation = nn.UpsamplingBilinear2d(size=tuple(cfg.bilinear_interpolation.output_size))
        self.training = True

        self.roi_head = get_roi_head(cfg)

        # self.proposal_generator = proposal_generator
        # self.roi_heads = roi_heads

        # self.input_format = input_format
        # self.vis_period = vis_period
        # if vis_period > 0:
        #     assert input_format is not None, "input_format is required for visualization!"

        # self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        # assert (
        #     self.pixel_mean.shape == self.pixel_std.shape
        # ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    def forward(self, images: torch.Tensor):
        """
        Input:
            images: tensor of shape (B, C, H, W)
        Output:
            pred_voxel: Voxel of shape (B, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
            refined_mesh:
                A list containing Mesh objects from three refinement stages
        """
        if not self.training:
            return self.inference(images)

        # Extract image features
        # keys: '0', '1', '2', '3'
        feature_dict = self.backbone(images)  

        # bilinear interpolation
        interpolated = self.bilinear_interpolation(feature_dict['3'])  # (B, 2048, 24, 24)
        feature_dict["interpolated"] = interpolated

        # obtain voxel, mesh prediction
        pred_voxel, refined_mesh = self.roi_head(feature_dict)
        return pred_voxel, refined_mesh

    def inference(self, images):
        """
        Input:
            images: tensor of shape (B, C, H, W)
        Output:
            pred_voxel: Voxel of shape (B, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
            refined_mesh:
                A list containing Mesh objects from three refinement stages
        """
        # Extract image features
        # keys: '0', '1', '2', '3'
        feature_dict = self.backbone(images)  

        # bilinear interpolation
        interpolated = self.bilinear_interpolation(feature_dict['3'])  # (B, 2048, 24, 24)
        feature_dict["interpolated"] = interpolated

        # obtain voxel, mesh prediction
        pred_voxel, refined_mesh = self.roi_head(feature_dict)
        return pred_voxel, refined_mesh

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    model = MeshRCNN(cfg)
    print(model)

    image = torch.zeros([1, 3, 224, 224])
    print(model(image))

if __name__ == "__main__":
    main()