import time
import matplotlib.pyplot as plt 
import numpy as np
import os
import logging
import sys 
from argparse import ArgumentParser
from collections import OrderedDict
from tqdm import tqdm
from logger import setup_logger
from collections import defaultdict

from omegaconf import DictConfig, OmegaConf
import hydra

import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.manifold import TSNE


from models import MeshRCNN
from dataset.dataset import get_dataloader
from losses import calculate_loss
from vis_utils import render_mesh

def get_tsne(feats):
    tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1))
    proj = tsne.fit_transform(feats)

    return proj

def plot_tsne(features, labels, title, save_path, legend=True):
    cmap = plt.cm.get_cmap('viridis', len(np.unique(labels))) # use a different color for each label

    for color, label in enumerate(np.unique(labels)):
        mask = labels==label
        nof_points = len(features[mask,0])

        if legend:
            scatter = plt.scatter(features[mask,0], features[mask,1], cmap=cmap, label=label)
        else:
            plt.scatter(features[mask,0], features[mask,1], cmap=cmap)

    # Add axis labels and a title
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)

    plt.close()


def tsne_visualization(cfg):
    ###
    # Setup Dataloader
    ###
    seen_dataset, seen_loader = get_dataloader(cfg.dataloader_seen)

    unseen_dataset, unseen_loader = get_dataloader(cfg.dataloader_unseen)

    ###
    # Load Model
    ###
    model = MeshRCNN(cfg)
    model.to(cfg.device)

    model.eval()

    # Load checkpoint
    print(f"Resuming from checkpoint {cfg.checkpoint_path}.")
    loaded_data = torch.load(cfg.checkpoint_path)

    # remove 'module.'
    new_state_dict = OrderedDict()
    for k, v in loaded_data["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


    # Obtain features for seen categories
    seen_features = []
    seen_labels = []
    for i, feed_dict in tqdm(enumerate(seen_loader)):
        images_gt, mesh_gt, voxel_gt = feed_dict["img"], feed_dict["mesh"], feed_dict["vox"]
        
        # e.g., /ssd1/chengyeh/Objaverse/data/horse/fea999c1d8d64eb4a9c2bb1fac1e7d83
        path_prefix = seen_dataset.get_path_prefix(i)
        seen_labels.append(path_prefix.split("/")[5])

        images_gt = images_gt.to(cfg.device)
        mesh_gt = mesh_gt.to(cfg.device)
        voxel_gt = voxel_gt.to(cfg.device)

        # obtain intermediate features
        _, refined_mesh, intermediate_feature = model(images_gt, intermediate_feature=True) # (B, 2048, 24, 24)
        intermediate_feature = intermediate_feature['interpolated']
        seen_features.append(intermediate_feature.flatten(1).detach().cpu().numpy())


        # plot the 3D mesh for this object
        render_mesh(refined_mesh[-1], os.path.join(cfg.vis_output_dir, "Seen",
                                                        path_prefix.split("/")[5], 
                                                        path_prefix.split("/")[6], 'Refined_Mesh_{}.gif'.format(cfg.roi_head.ROI_MESH_HEAD.NUM_STAGES)), 
                                                        cfg)
        render_mesh(mesh_gt, os.path.join(cfg.vis_output_dir, "Seen",
                                                        path_prefix.split("/")[5], 
                                                        path_prefix.split("/")[6], 'GT_Mesh.gif'), 
                                                        cfg)
    seen_features = np.concatenate(seen_features, axis=0)

    # Obtain features for Unseen categories
    unseen_features = []
    unseen_labels = []
    for i, feed_dict in tqdm(enumerate(unseen_loader)):
        images_gt, mesh_gt, voxel_gt = feed_dict["img"], feed_dict["mesh"], feed_dict["vox"]
        path_prefix = unseen_dataset.get_path_prefix(i)
        unseen_labels.append(path_prefix.split("/")[5])

        images_gt = images_gt.to(cfg.device)
        mesh_gt = mesh_gt.to(cfg.device)
        voxel_gt = voxel_gt.to(cfg.device)

        # obtain intermediate features
        _, refined_mesh, intermediate_feature = model(images_gt, intermediate_feature=True) # (B, 2048, 24, 24)
        intermediate_feature = intermediate_feature['interpolated']


        unseen_features.append(intermediate_feature.flatten(1).detach().cpu().numpy())
        
        # plot mesh
        render_mesh(refined_mesh[-1], os.path.join(cfg.vis_output_dir, "Unseen",
                                                        path_prefix.split("/")[5], 
                                                        path_prefix.split("/")[6], 'Refined_Mesh_{}.gif'.format(cfg.roi_head.ROI_MESH_HEAD.NUM_STAGES)), 
                                                        cfg)
        render_mesh(mesh_gt, os.path.join(cfg.vis_output_dir, "Unseen",
                                                        path_prefix.split("/")[5], 
                                                        path_prefix.split("/")[6], 'GT_Mesh.gif'), 
                                                        cfg)
    unseen_features = np.concatenate(unseen_features, axis=0)

    # Plot tsne for seen vs unseen
    projected_features = get_tsne(np.concatenate([seen_features, unseen_features], axis=0))
    seen_unseen_labels = np.array(["Seen"] * len(seen_features) + ['Unseen'] * len(unseen_features))
    plot_tsne(projected_features, seen_unseen_labels, "Seen vs Unseen TSNE", os.path.join(cfg.output_dir, 'Seen_UnSeen_TSNE.png'))

    # plot tsne for all labels
    categories = np.array(seen_labels + unseen_labels)
    # plot_tsne(projected_features, categories, "All Categories TSNE", os.path.join(cfg.output_dir, 'All_Category_TSNE.png'))
    # plot_tsne(projected_features, categories, "All Categories TSNE", os.path.join(cfg.output_dir, 'All_Category_WO_Legend_TSNE.png'), legend=False)


@hydra.main(version_base=None, config_path="../configs", config_name="visualization")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.vis_output_dir, exist_ok=True)
    tsne_visualization(cfg)

if __name__ == '__main__':
    main()