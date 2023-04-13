import torch
import torch.nn as nn

from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes



# TODO:
def calculate_loss(images_gt, mesh_gt, voxel_gt, pred_voxel, refined_mesh, cfg):

	v_loss = voxel_loss(pred_voxel, voxel_gt)

	sample_trg, sample_trg_normals = sample_points_from_meshes(mesh_gt, 
							    num_samples=cfg.PRED_NUM_SAMPLES, 
								return_normals=True)
	sample_pred, sample_pred_normals = sample_points_from_meshes(refined_mesh, 
							      num_samples=cfg.PRED_NUM_SAMPLES,
								  return_normals=True)

	c_loss, n_loss = chamfer_loss(
				sample_pred, sample_trg, 
		    	x_normals=sample_pred_normals,
				y_normals=sample_trg_normals
			)

	e_loss = mesh_edge_loss(sample_pred, sample_trg)

	# l_loss = smoothness_loss(sample_pred)

	loss = cfg.CHAMFER_LOSS_WEIGHT * c_loss \
		+ cfg.NORMALS_LOSS_WEIGHT * n_loss \
		+ cfg.EDGE_LOSS_WEIGHT * e_loss \
		+ cfg.VOXEL_LOSS_WEIGHT * v_loss

	return loss





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



def normal_distance(pred_mesh, target_mesh):
	norm_p = mesh_normal_consistency(pred_mesh)
	norm_t = mesh_normal_consistency(target_mesh)

	loss = 0.5 * (norm_p + norm_t)

	return loss
