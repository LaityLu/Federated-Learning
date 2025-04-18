import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from .logger_setup import setup_logger

logger = setup_logger()


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


def visual_pca_whitening(data, client_idxes, the_round):
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

    for i, idx in enumerate(client_idxes):
        ax.text(data_w[i, 0], data_w[i, 1], data_w[i, 2], idx, fontsize=8, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(f'./save/fig/{the_round}.png', bbox_inches='tight')


def visual_cluster(data, client_idxes, cluster, the_round):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map for clusters
    # unique_clusters = set(cluster)
    color_map = {-1: 'red', 0: 'blue'}

    # Plot each cluster
    plotted_clusters = set()
    for i, (point, idx) in enumerate(zip(data, cluster)):
        color = color_map[idx]
        label = None

        # Only add label once per cluster
        if idx not in plotted_clusters:
            label = f'Cluster {idx}' if idx != -1 else 'Noise Points'
            plotted_clusters.add(idx)

        ax.scatter(point[0], point[1], point[2],
                   c=[color], label=label, s=50, alpha=0.7)

        # Add client ID text
        ax.text(point[0], point[1], point[2], str(client_idxes[i]),
                fontsize=8, ha='center', va='bottom')

    # Origin point
    ax.scatter(0, 0, 0, c='black', s=30, marker='*', label='Origin')

    # Styling
    ax.set_title(f'Cluster Results (Round {the_round})', fontsize=14, pad=20)
    ax.set_xlabel('Cosine Similarity', fontsize=10, labelpad=10)
    ax.set_ylabel('Manhattan Distance', fontsize=10, labelpad=10)
    ax.set_zlabel('Euclidean Distance', fontsize=10, labelpad=10)

    # Adjust legend and layout
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save with metadata
    plt.tight_layout()
    save_path = f'./save/fig/cluster/cluster_round_{the_round}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def custom_serializer(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def check(config_path, config):
    # check existing dataset
    if os.path.exists(config_path):
        with open(f'{config_path}', 'r') as f:
            config1 = json.load(f)
        if config1['num_clients'] == config['FL']['num_clients'] and \
                config1['Dataset'] == config['Dataset'] and \
                config1['Sampler'] == config['Sampler']:
            logger.info("Dataset already sampled.")
            return True
    return False


def save_clients_data(config_path, config, dict_clients, list_num_dps):
    config1 = {
        'num_clients': config['FL']['num_clients'],
        'Dataset': config['Dataset'],
        'Sampler': config['Sampler']
    }
    dict_clients_data = {
        client_id: {
            'dps_idx': list(dict_clients[client_id]),
            'num_dps': list_num_dps[client_id]
        }
        for client_id in range(len(list_num_dps))
    }
    with open(config_path + '/data_config.json', 'w') as f:
        json.dump(config1, f)
    with open(config_path + '/dps.json', 'w') as f:
        json.dump(dict_clients_data, f, default=custom_serializer)
    logger.info("The sampled dataset has been saved to disk.")


def load_clients_data(config_path):
    with open(config_path + '/dps.json', 'r') as f:
        dict_clients_data = json.load(f)
    dict_clients = {}
    list_num_dps = []
    for client_id, data in dict_clients_data.items():
        dict_clients[int(client_id)] = data['dps_idx']
        list_num_dps.append(data['num_dps'])
    logger.info("The client's data has been loaded.")
    return dict_clients, list_num_dps


def save_select_info(path, info):
    with open(path + '/select_info.json', 'w') as f:
        json.dump(info, f)
    logger.info("The selected clients info has been saved to disk.")


def load_select_info(path):
    with open(path + '/select_info.json', 'r') as f:
        select_info = json.load(f)
    logger.info("The select clients info has been loaded.")
    return select_info


def save_train_loss(path, info):
    with open(path + '/train_loss.json', 'w') as f:
        json.dump(info, f)
    logger.info("The train loss info has been saved to disk.")


def load_train_loss(path):
    with open(path + '/train_loss.json', 'r') as f:
        train_loss = json.load(f)
    logger.info("The train loss info has been loaded.")
    return train_loss


def save_global_model(dataset, global_round, global_model):
    model_path = os.path.join("./save/historical_information", dataset, 'server_models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, "round_" + str(global_round) + ".pth")
    torch.save(global_model.state_dict(), model_path)


def load_global_model(dataset, global_round):
    model_path = os.path.join("./save/historical_information", dataset, 'server_models')
    model_path = os.path.join(model_path, "round_" + str(global_round) + ".pth")
    assert (os.path.exists(model_path)), "No saved global model"
    return model_path


def save_client_model(dataset, global_round, client_model_state_dict):
    model_path = os.path.join("./save/historical_information", dataset, 'client_models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, "round_" + str(global_round) + ".pth")
    torch.save(client_model_state_dict, model_path)


def load_client_model(dataset, global_round):
    model_path = os.path.join("./save/historical_information", dataset, 'client_models')
    model_path = os.path.join(model_path, "round_" + str(global_round) + ".pth")
    assert (os.path.exists(model_path)), "No saved client model"
    return model_path


if __name__ == '__main__':
    pass
