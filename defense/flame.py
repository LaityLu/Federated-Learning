import copy

import numpy as np
import torch
from sklearn.cluster import HDBSCAN

import aggregator
from utils import parameters_dict_to_vector
from utils.logger_config import logger


class Flame:

    def __init__(self, noise=0.001):
        self.noise = noise

    def exec(self, global_model, client_models, client_idxes, num_dps, aggregator_name):
        num_clients = len(client_models)
        # compute the update for every client
        update_params = []
        for i in range(num_clients):
            update = {}
            for key, var in global_model.items():
                update[key] = client_models[i][key] - global_model[key]
            update_params.append(update)
        # flatten the model into a one-dimensional tensor
        v_update_params = [parameters_dict_to_vector(up) for up in update_params]
        v_client_models = [parameters_dict_to_vector(cm) for cm in client_models]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        cos_list = []
        for i in range(len(v_client_models)):
            cos_i = []
            for j in range(len(v_client_models)):
                cos_ij = 1 - cos(v_client_models[i], v_client_models[j])
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)
        cluster = HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

        benign_client = []
        malicious_client = client_idxes.copy()
        norm_list = np.array([])  # euclidean distance
        max_num_in_cluster = 0
        max_cluster_index = 0
        bgn_idx = []
        cluster_labels = cluster.labels_
        # all clients are benign
        if cluster_labels.max() < 0:
            for i in range(num_clients):
                benign_client.append(client_idxes[i])
        else:
            # find clients in the largest cluster and regard them as benign_clients
            for index_cluster in range(cluster_labels.max() + 1):
                if len(cluster_labels[cluster_labels == index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(cluster_labels[cluster_labels == index_cluster])
            for i in range(num_clients):
                if cluster_labels[i] == max_cluster_index:
                    bgn_idx.append(i)
                    benign_client.append(client_idxes[i])
                    malicious_client.remove(client_idxes[i])
        for i in range(num_clients):
            norm_list = np.append(norm_list, torch.norm(v_update_params[i], p=2).item())

        logger.info("cluster labels: {}".format(cluster_labels))
        logger.info("The benign clients: {}".format(benign_client))
        logger.info("The malicious clients: {}".format(malicious_client))

        clip_value = np.median(norm_list)
        for i in range(len(benign_client)):
            gama = clip_value / norm_list[i]
            if gama < 1:
                for key in update_params[i]:
                    if key.split('.')[-1] == 'num_batches_tracked':
                        continue
                    update_params[i][key] *= gama

        # aggregation
        selected_models = [client_models[i] for i in bgn_idx]
        selected_num_dps = [num_dps[i] for i in bgn_idx]
        global_model_state_dict = getattr(aggregator, aggregator_name)(selected_models, selected_num_dps)

        # add noise
        for key, var in global_model_state_dict.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            temp = copy.deepcopy(var)
            temp = temp.normal_(mean=0, std=self.noise * clip_value)
            var += temp
        return global_model_state_dict
