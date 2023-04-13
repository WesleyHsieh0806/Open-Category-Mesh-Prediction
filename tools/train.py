import time
import matplotlib.pyplot as plt 
import numpy as np
import os

from omegaconf import DictConfig, OmegaConf
import hydra


import torch


from models import MeshRCNN
from dataset.dataset import get_dataloader
from .losses import calculate_loss


# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ["HYDRA_FULL_ERROR"] = "1"



def train_model(cfg):
    obj_dataset, loader = get_dataloader(cfg.dataloader) 
    train_loader = iter(loader)

    model = MeshRCNN(cfg)

    model.to(cfg.training.device)
    model.train()



    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.training.lr) 
    start_iter = 1
    start_time = time.time()

    checkpoint_path = cfg.training.checkpoint_path
    if len(checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    losses = []

    print("Starting training !")
    for step in range(start_iter, cfg.training.max_iter+1):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, mesh_gt, voxel_gt = feed_dict["img"], feed_dict["mesh"], feed_dict["vox"]
        read_time = time.time() - read_start_time

        images_gt = images_gt.to(cfg.training.device)
        mesh_gt = mesh_gt.to(cfg.training.device)
        voxel_gt = voxel_gt.to(cfg.training.device)

        pred_voxel, refined_mesh = model(images_gt)

        v_loss, c_loss, n_loss, e_loss = calculate_loss(images_gt, mesh_gt, voxel_gt, pred_voxel, refined_mesh, cfg.roi_head.ROI_MESH_HEAD)

        loss = cfg.roi_head.ROI_MESH_HEAD.CHAMFER_LOSS_WEIGHT * c_loss \
            + cfg.roi_head.ROI_MESH_HEAD.NORMALS_LOSS_WEIGHT * n_loss \
            + cfg.roi_head.ROI_MESH_HEAD.EDGE_LOSS_WEIGHT * e_loss \
            + cfg.roi_head.ROI_VOXEL_HEAD.VOXEL_LOSS_WEIGHT * v_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        # Checkpoint.
        if (
            step % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }

            torch.save(data_to_store, checkpoint_path)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, cfg.training.max_iter, total_time, read_time, iter_time, loss_vis))

        losses.append(loss_vis)
    print('Done!')

    plt.plot(np.arange(cfg.training.max_iter), losses, marker='o')
    plt.savefig(f'train_loss_baseline.png')

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    train_model(cfg)

if __name__ == '__main__':
    main()