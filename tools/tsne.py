import torch
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
from utils import *
import matplotlib.pyplot as plt
from matplotlib import patches
from train_q2 import ResNet

def get_feats(model, test_loader, n_class):
    #TODO: return_nodes, where to get the features
    return_nodes = # return_nodes = {'resnet': 'avgpool'}
    truncated_model = create_feature_extractor(model, return_nodes=return_nodes)
    feats = []
    targets = []
    n = 0
    for data, target, _ in test_loader:
        # feat = truncated_model(data.to("cuda"))['avgpool'].view((data.shape[0], -1))
        #TODO
        feat = truncated_model(data.to("cuda"))[''].view((data.shape[0], -1))
        feats.append(feat.detach().cpu().numpy())
        #TODO
        targets.append(target.view(-1, n_class).detach().cpu().numpy())
        n += 1
        #TODO
        if n == :
        # if n == 10:
            break
    return  np.concatenate(feats), np.concatenate(targets).astype(np.int32)

def get_tsne(feats):
    tsne = TSNE()
    proj = tsne.fit_transform(feats)

    return proj

def plot_exp1(proj, targets, n_class=2):
    colors = [[0, 0, 255], [255, 0, 0]]
    plt.figure(figsize=(12,10))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.array(colors)/255)
    plt.legend(handles=[patches.Patch(color=np.array(colors[i])/255, label="class " + str(i)) for i in range(n_class)])
    plt.title("tsne_seleceted_vs_unseen")
    plt.savefig("tsne_seleceted_vs_unseen.png")

def plot_exp2(proj, targets, n_class=10):
    colors = np.array([[np.random.choice(np.arange(256), size=3)] for i in range(n_class)])

    mean_colors = []
    for i in range(proj.shape[0]):
        colors1 = colors[np.where(targets[i, :]==1)]
        mean_colors.append(np.mean(colors1, axis=0, dtype=np.int32))

    plt.figure(figsize=(12,10))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.array(mean_colors)/255)
    plt.legend(handles=[patches.Patch(color=np.array(colors[i])/255, label="class " + str(i)) for i in range(n_class)])
    plt.title("tsne_shift_between_categories")
    plt.savefig("tsne_shift_between_categories.png")


if __name__ == '__main__':
    #TODO: model
    model_path = ""  
    model = .to('cuda')
    model = torch.load(model_path)
    model.eval()

    test_loader = get_data_loader('voc', train=False, batch_size=100, split='test')

    # experiment 1: n_class = 2 (selected categories and unseen categories: blue / red)
    feats, targets = get_feats(model, test_loader, n_class=2)

    proj = get_tsne(feats)
    
    plot_exp1(proj, targets, n_class=2)

    # experiment 2: n_class = 10 (shift between categories: 10 colors)
    feats, targets = get_feats(model, test_loader, n_class=10)

    proj = get_tsne(feats)
    
    plot_exp2(proj, targets, n_class=10)
