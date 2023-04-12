import time
import torch
from models import MeshRCNN
from dataset.dataset import ObjaverseDataset
from  import calculate_loss #??
import matplotlib.pyplot as plt 
import numpy as np


def train_model(cfg):
    obj_dataset = ObjaverseDataset(cfg.dataloader) #?? can use
    loader = torch.utils.data.DataLoader(
        obj_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model = MeshRCNN(cfg)
    model.to(cfg.training.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.training.lr) 
    start_epoch = 0
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
    for step in range(start_epoch, cfg.training.num_epochs):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, mesh_gt, voxel_gt = feed_dict # 3 meshes and voxels?
        read_time = time.time() - read_start_time

        pred_voxel, refined_mesh = model(images_gt)

        loss = calculate_loss(images_gt, mesh_gt, voxel_gt, pred_voxel, refined_mesh)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

        losses.append(loss_vis)
    print('Done!')

    plt.plot(np.arange(args.max_iter), losses, marker='o')
    plt.savefig(f'loss_{args.type}.png')

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    train_model(cfg)

if __name__ == '__main__':
    main()