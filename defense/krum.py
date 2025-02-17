import numpy as np
import torch

import aggregator
from utils.logger_config import logger


class Krum:
    # if num_selected_clients = 1, it's Krum.
    # if num_selected_clients = n - f, it's Multi-Krum.
    def __init__(self, estimated_num_attacker, num_selected_clients=1):
        self.num_adv = estimated_num_attacker
        self.num_selected = num_selected_clients

    def exec(self, global_model, client_models, client_idxes, num_dps, aggregator_name):

        # flatten the model into a one-dimensional tensor
        v_client_models = [torch.cat([p.view(-1) for p in cm.values()]).detach().cpu().numpy() for cm in client_models]

        # compute the distance between different clients
        num_clients = len(client_models)
        dist_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = float(np.linalg.norm(v_client_models[i] - v_client_models[j]) ** 2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # compute sum_dist and choose the client with minimum as benign client
        scores = []
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - self.num_adv)]])
            scores.append(sum_dist)
        sorted_list = np.argpartition(scores, self.num_selected)
        selected_index = sorted_list[:self.num_selected]
        non_selected_index = sorted_list[self.num_selected:]

        # store the idxes and scores of benign clients and malicious clients
        benign_clients = [client_idxes[i] for i in selected_index]
        benign_scores = [scores[i].round(2) for i in selected_index]
        adv_clients = [client_idxes[i] for i in non_selected_index]
        adv_scores = [scores[i].round(2) for i in non_selected_index]
        logger.info('clients idxes:{}'.format(client_idxes))
        logger.info("The benign clients: {},\n\t scores:{}".format(benign_clients, benign_scores))
        logger.info("The malicious clients: {},\n\t scores:{}".format(adv_clients, adv_scores))

        # aggregation
        selected_models = [client_models[i] for i in selected_index]
        selected_num_dps = [num_dps[i] for i in selected_index]
        global_model_state_dict = getattr(aggregator, aggregator_name)(selected_models, selected_num_dps)

        # return the aggregated global model state dict
        return global_model_state_dict
