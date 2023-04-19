import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F


class VoxelRCNNConvUpsampleHead(nn.Module):
    """
    A voxel head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape):
        super(VoxelRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        self.num_classes   = 1 if (cfg.roi_head.ROI_VOXEL_HEAD.CLASS_AGNOSTIC) else cfg.roi_head.NUM_CLASSES
        conv_dims          = cfg.roi_head.ROI_VOXEL_HEAD.CONV_DIM
        input_channels     = input_shape[0]  # 2048
        # fmt: on

        self.num_depth = cfg.roi_head.ROI_VOXEL_HEAD.VOXEL_SIZE


        self.up_sample_conv = nn.Sequential(
                    nn.Conv2d(input_channels, conv_dims, kernel_size=(3, 3), stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_dims, conv_dims, kernel_size=(3, 3), stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(conv_dims, conv_dims, kernel_size=(2, 2), stride=2),
                    nn.ReLU(inplace=True),
        )

        self.predictor = nn.Conv2d(conv_dims, self.num_depth, kernel_size=1, stride=1)
        
        def init_weight(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                weight_init.c2_msra_fill(module)

        # # initialize weights
        # self.up_sample_conv.apply(init_weight) 

        # # use normal distribution initialization for voxel prediction layer
        # nn.init.normal_(self.predictor.weight, std=0.001)


    def forward(self, x):
        '''
        * Input: feature map of shape (B, C, H, W)
        * Output: Voxel of shape (B, 1, NUM_DEPTH, 2C, 2H)
        '''
        x = self.up_sample_conv(x)
        voxel = self.predictor(x)

        # reshape from (N, CD, H, W) to (N, C, D, H, W)
        voxel = voxel.reshape(voxel.size(0), self.num_classes, self.num_depth, voxel.size(2), voxel.size(3))
        return voxel


def build_voxel_head(cfg, input_shape):
    return VoxelRCNNConvUpsampleHead(cfg, input_shape)