from typing import Dict
import torch

from omegaconf import DictConfig, OmegaConf
import hydra

from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from torch import nn

from models.roi_head.mesh_head import (
    build_mesh_head,
#     mesh_rcnn_inference,
#     mesh_rcnn_loss,
)
from models.roi_head.voxel_head import (
    build_voxel_head,
    # voxel_rcnn_inference,
    # voxel_rcnn_loss,
)


class MeshRCNNROIHeads(nn.Module):
    """
    The ROI specific heads for Mesh R-CNN
    """

    def __init__(self, cfg):
        super().__init__()
        self._init_voxel_head(cfg)
        self._init_mesh_head(cfg)



    def _init_voxel_head(self, cfg):
        input_shape = cfg.roi_head.ROI_VOXEL_HEAD.input_shape
        # fmt: on

        self.voxel_loss_weight = cfg.roi_head.ROI_VOXEL_HEAD.voxel_loss_weight
        self.cubify_thresh = cfg.roi_head.ROI_VOXEL_HEAD.CUBIFY_THRESH

        self.voxel_head = build_voxel_head(cfg, input_shape)
        # attribute setup
        self.cls_agnostic_voxel = cfg.roi_head.ROI_VOXEL_HEAD.CLASS_AGNOSTIC

    def _init_mesh_head(self, cfg):

        self.chamfer_loss_weight = cfg.roi_head.ROI_MESH_HEAD.CHAMFER_LOSS_WEIGHT
        self.normals_loss_weight = cfg.roi_head.ROI_MESH_HEAD.NORMALS_LOSS_WEIGHT
        self.edge_loss_weight = cfg.roi_head.ROI_MESH_HEAD.EDGE_LOSS_WEIGHT
        self.gt_num_samples = cfg.roi_head.ROI_MESH_HEAD.GT_NUM_SAMPLES
        self.pred_num_samples = cfg.roi_head.ROI_MESH_HEAD.PRED_NUM_SAMPLES
        self.gt_coord_thresh = cfg.roi_head.ROI_MESH_HEAD.GT_COORD_THRESH


        self.mesh_head = build_mesh_head(cfg)
    def forward_voxel(self, feature_dict):
        """
        Input: 
            feature_dict contains the following keys
                0: conv2_3 features
                1: conv3_4 features
                2: conv4_6 features
                3: conv5_3 features
                interpolated: bilinear interpolated features of shape (2048, cfg.bilinear_interpolation.output_size, cfg.bilinear_interpolation.output_size)
        Output:
            Coarse Voxel (B, 1, V, V, V)
        """
        conv2_3, conv3_4, conv4_6, conv5_3 = feature_dict['0'], feature_dict['1'], feature_dict['2'], feature_dict['3']
        bi_interp_features = feature_dict["interpolated"]

        pred_voxel = self.voxel_head(bi_interp_features)
        return pred_voxel

    def forward(self, feature_dict):
        """
        Input: 
            feature_dict contains the following keys
                0: conv2_3 features
                1: conv3_4 features
                2: conv4_6 features
                3: conv5_3 features
                interpolated: bilinear interpolated features of shape (2048, cfg.bilinear_interpolation.output_size, cfg.bilinear_interpolation.output_size)
        Output:
            Voxel, Mesh(A list containing Mesh objects from three refinement stages)
        """
        # predict voxels with extracted image features
        pred_voxel = self.forward_voxel(feature_dict)

        # generate init mesh with cubify
        if self.cls_agnostic_voxel:
            with torch.no_grad():
                vox_in = pred_voxel.sigmoid().squeeze(1)  # (N, V, V, V)
                init_mesh = cubify(vox_in, self.cubify_thresh)  # 1
        else:
            raise ValueError("No support for class specific predictions")

        # Obtain Refined Meshes from three refinement stages
        refined_mesh = self.mesh_head(feature_dict, init_mesh)  # [refined_meshes_from_stage1, refined_meshes_from_stage2, refined_meshes_from_stage3]
        return pred_voxel, refined_mesh



        
get_roi_head_by_name = {
    "MeshRCNNROIHeads": lambda cfg: MeshRCNNROIHeads(cfg),
}

def get_roi_head(cfg):
    return get_roi_head_by_name[cfg.roi_head.type](cfg)
    

@hydra.main(version_base=None, config_path="../../configs", config_name="baseline")
def main(cfg: DictConfig):
    model = MeshRCNNROIHeads(cfg)
    print(model)

    image_feature = torch.zeros([1, 2048, 7, 7])
    # print(model.backbone(image).shape)

if __name__ == "__main__":
    main()