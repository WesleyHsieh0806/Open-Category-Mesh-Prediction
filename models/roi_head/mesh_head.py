from torch import nn
from typing import List
class MeshHead(nn.Module):
    def __init__(self):
        pass


from collections import OrderedDict
import fvcore.nn.weight_init as weight_init
import torch
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import GraphConv, SubdivideMeshes, sample_points_from_meshes, vert_align
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F



class ResGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, gconv_init="normal", directed=False):
        '''
        ResGraphConv consists of two graphconv layer and one linear projection layer as skipped connection
              x (input_dim)------|
              |                  |
           ResGraph (output_dim) MLP
              |                  |
           ResGraph (output_dim) x_res
              |-----> + <--------| 
                      |
                    output
        '''
        super().__init__()
        self.graph_conv1 = GraphConv(input_dim, output_dim, init=gconv_init, directed=directed)  # very similar to linear layer
        self.graph_conv2 = GraphConv(output_dim, output_dim, init=gconv_init, directed=directed)  # very similar to linear layer
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x, edge):
        '''
        * x: tensor of shape (B*V, input_dim)
        * edge: tensor of shape (B*E, 2)
            E denotes number of edges
        * Output:
            tensor of shape (B*V, output_dim)
        '''
        h1 = self.graph_conv1(F.relu(x), edge)
        h2 = self.graph_conv2(F.relu(h1), edge)

        skipped_x = x if (self.skip_proj is None) else self.skip_proj(x)
        return h2 + skipped_x

class MeshRefinementStage(nn.Module):
    def __init__(self, img_channles: List[int], vert_feat_dim, hidden_dim, num_res_graph_convs, gconv_init="normal"):
        """
        Args:
          img_channels: Dimension of resnet features we will get from vert_align
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          num_res_graph_convs: number of residual graph conv layers
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        # fc layer to reduce feature dimension
        self.bottleneck = nn.Linear(sum(img_channles), hidden_dim)  # 3840 -> 128

        # Residual Graph Convolution Layer
        self.res_gconvs = nn.ModuleList()
        for i in range(num_res_graph_convs):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim
            res_graph_conv = ResGraphConv(input_dim, hidden_dim, gconv_init=gconv_init, directed=False)  # 259 -> 128
            self.res_gconvs.append(res_graph_conv)  # 259 -> 128 -> 128 -> 128

        # one graph convs
        self.gconv = GraphConv(hidden_dim, 3, init=gconv_init, directed=False)

        # initialization
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)


    def forward(self, feature_dict, mesh, vert_feats=None):
        '''
        Input: 
            feature_dict contains the following keys
                0: conv2_3 features
                1: conv3_4 features
                2: conv4_6 features
                3: conv5_3 features
                interpolated: bilinear interpolated features of shape (2048, cfg.bilinear_interpolation.output_size, cfg.bilinear_interpolation.output_size)
            mesh: (coarse mesh predicted by voxel_head and cubify)
        Output:
            refined_Mesh after 1 refinement stage
        '''
        conv2_3, conv3_4, conv4_6, conv5_3 = feature_dict['0'], feature_dict['1'], feature_dict['2'], feature_dict['3']
        img_feats = torch.cat([vert_align(conv2_3, mesh, return_packed=True, padding_mode="border"),
                                vert_align(conv3_4, mesh, return_packed=True, padding_mode="border"),
                                vert_align(conv4_6, mesh, return_packed=True, padding_mode="border"),
                                vert_align(conv5_3, mesh, return_packed=True, padding_mode="border")], dim=1)  # (B*V, 3840)
        # 3840 -> hidden_dim 
        img_feats = F.relu(self.bottleneck(img_feats))
        if vert_feats is None:
            # hidden_dim + 3
            vert_feats = torch.cat((img_feats, mesh.verts_packed()), dim=1)  # (B*V, hidden_dim + 3)
        else:
            # hidden_dim * 2 + 3
            vert_feats = torch.cat((vert_feats, img_feats, mesh.verts_packed()), dim=1)

        # Three residual graph conv layers: output will be new vert_feats  # B*V, hidden_dim
        for res_graph_conv in self.res_gconvs:
            vert_feats = res_graph_conv(vert_feats, mesh.edges_packed()) 

        # graph conv layers
        pre_offset_feat = self.gconv(vert_feats, mesh.edges_packed())

        # refine vertex positions
        deform = torch.tanh(pre_offset_feat)
        mesh = mesh.offset_verts(deform)
        return mesh, vert_feats


class MeshRCNNGraphConvHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine layers.
    """

    def __init__(self, cfg):
        super(MeshRCNNGraphConvHead, self).__init__()

        # fmt: off
        num_stages           = cfg.roi_head.ROI_MESH_HEAD.NUM_STAGES
        num_res_graph_convs  = cfg.roi_head.ROI_MESH_HEAD.NUM_RES_GRAPH_CONVS  # per stage
        graph_conv_dim       = cfg.roi_head.ROI_MESH_HEAD.GRAPH_CONV_DIM
        graph_conv_init      = cfg.roi_head.ROI_MESH_HEAD.GRAPH_CONV_INIT
        input_channels       = cfg.roi_head.ROI_MESH_HEAD.INPUT_CHANNELS
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            vert_feat_dim = 0 if i == 0 else graph_conv_dim
            stage = MeshRefinementStage(
                input_channels,
                vert_feat_dim,
                graph_conv_dim,
                num_res_graph_convs,
                gconv_init=graph_conv_init,
            )
            self.stages.append(stage)

    def forward(self, feature_dict, mesh):
        '''
        Input: 
            feature_dict contains the following keys
                0: conv2_3 features
                1: conv3_4 features
                2: conv4_6 features
                3: conv5_3 features
                interpolated: bilinear interpolated features of shape (2048, cfg.bilinear_interpolation.output_size, cfg.bilinear_interpolation.output_size)
            mesh: (coarse mesh predicted by voxel_head and cubify)
        Output:
            refined meshes output from all refinmentstage
        '''
        if mesh.isempty():
            return [Meshes(verts=[], faces=[])]

        meshes = []
        vert_feats = None
        for stage in self.stages:
            mesh, vert_feats = stage(feature_dict, mesh, vert_feats=vert_feats)
            meshes.append(mesh)
        return meshes




def build_mesh_head(cfg):
    return MeshRCNNGraphConvHead(cfg)