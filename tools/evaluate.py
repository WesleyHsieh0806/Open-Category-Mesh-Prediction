import time
import torch
from models import MeshRCNN
from dataset.dataset import get_dataloader
from  import evaluate #??
import matplotlib.pyplot as plt 
import numpy as np

def save_plot(thresholds, avg_f1_score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation')
    plt.savefig(f'eval_baseline', bbox_inches='tight')

def evaluate_model(cfg):
    obj_dataset = get_dataloader(cfg.dataloader_test) #?? can use
    loader = torch.utils.data.DataLoader(
        obj_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = MeshRCNN(cfg)
    model.to(cfg.training.device)
    model.eval()

    start_epoch = 0
    start_time = time.time()

    thresholds = [0.1, 0.3, 0.5]
    avg_f1_score_01 = []
    avg_f1_score_03 = []
    avg_f1_score_05 = []
    avg_f1_score = []
    avg_chamf_dist = []

    checkpoint_path = cfg.training.checkpoint_path
    loaded_data = torch.load(checkpoint_path)
    model.load_state_dict(loaded_data["model"])
    
    print("Starting evaluating !")
    max_epoch = len(eval_loader)
    for step in range(start_epoch, max_epoch):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt, voxel_gt = feed_dict

        read_time = time.time() - read_start_time

        pred_voxel, refined_mesh = model(images_gt)

        metrics = evaluate(images_gt, mesh_gt, voxel_gt, pred_voxel, refined_mesh)

        img_step = step % 50 
        num = step // 50 
        if img_step == 0:
            render_mesh(predictions, output_path=f'images/q_2-3-pred-{num}.gif')
            render_mesh(mesh_gt.to(cfg.training.device), output_path=f'images/q_2-3-gt-{num}.gif')

      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_01, f1_03, f1_05= metrics['F1@0.10000'], metrics['F1@0.30000'], metrics['F1@0.50000']
        avg_f1_score_01.append(f1_01)
        avg_f1_score_03.append(f1_03)
        avg_f1_score_05.append(f1_05)
        avg_chamf_dist.append(torch.tensor([metrics["Chamf_dist@%f"]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.1: %.3f; F1@0.3: %.3f; F1@0.5: %.3f; Avg F1@0.5: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_01, f1_03, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score)
    print('Done!')

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    evaluate_model(cfg)

if __name__ == '__main__':
    main()