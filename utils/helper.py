import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxes):
        self.dataset = dataset
        self.idxes = list(idxes)

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxes[item]]
        return image, label


def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def visual_pca_whitening(data, client_idxes, round):
    data_D = data - data.mean(axis=0)
    data_C = np.cov(data.T)
    data_value, data_vector = np.linalg.eig(data_C)
    data_r = np.dot(data_D, data_vector)
    data_w = data_r / np.sqrt(data_value)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_w[:, 0], data_w[:, 1], data_w[:, 2], c='b', label='Data Points')
    ax.scatter(0, 0, 0, c='r', s=10, label='Origin')

    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('cos‘')
    ax.set_ylabel('mht’')
    ax.set_zlabel('euc‘')

    for i, id in enumerate(client_idxes):
        ax.text(data_w[i, 0], data_w[i, 1], data_w[i, 2], id, fontsize=8, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(f'./save/fig/{round}.png', bbox_inches='tight')
