import torch
import torch.nn as nn
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance



# define losses
def voxel_loss(voxel_src, voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = nn.BCELoss()
	loss = nn.BCEWithLogitsLoss()
	# implement some loss for binary voxel grids
	prob_loss = loss(voxel_src, voxel_tgt)
	return prob_loss


def chamfer_loss(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	loss_chamfer = chamfer_distance(point_cloud_src, point_cloud_tgt, 
		batch_reduction="mean", point_reduction="sum")[0]

	return loss_chamfer



def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian