import torch
import numpy as np
from utils.logger_config import logger
import aggregator


class MultiMetrics:

    def __init__(self, top_k: float):
        self.top_k = top_k

    def exec(self, global_model, client_models, client_idxes, num_dps, aggregator_name):
        # flatten the model into a one-dimensional tensor
        v_global_model = torch.cat([p.view(-1) for p in global_model.values()]).detach().cpu().numpy()
        v_client_models = [torch.cat([p.view(-1) for p in cm.values()]).detach().cpu().numpy() for cm in client_models]

        num_clients = len(client_idxes)

        # store the distance between each client model and global model
        cos_dis = [0.0] * num_clients  # cosine distance
        euc_dis = [0.0] * num_clients  # euclidean distance
        mht_dis = [0.0] * num_clients  # manhattan distance
        for i, m_i in enumerate(v_client_models):
            # Compute the different value of cosine distance
            cosine_distance = float(
                (1 - np.dot(m_i, v_global_model) / (np.linalg.norm(m_i) * np.linalg.norm(
                    v_global_model))) ** 2)
            # Compute the different value of Manhattan distance
            manhattan_distance = float(np.linalg.norm(m_i - v_global_model, ord=1))
            # Compute the different value of Euclidean distance
            euclidean_distance = float(np.linalg.norm(m_i - v_global_model))
            cos_dis[i] += cosine_distance
            euc_dis[i] += euclidean_distance
            mht_dis[i] += manhattan_distance

        # store the difference between different client models at three distance
        cos_dd = [0.0] * num_clients
        euc_dd = [0.0] * num_clients
        mht_dd = [0.0] * num_clients
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    c_dd = np.abs(cos_dis[i] - cos_dis[j])
                    e_dd = np.abs(euc_dis[i] - euc_dis[j])
                    m_dd = np.abs(mht_dis[i] - mht_dis[j])
                    cos_dd[i] += c_dd
                    euc_dd[i] += e_dd
                    mht_dd[i] += m_dd

        # combine into a matrix
        tri_distance = np.vstack([cos_dd, mht_dd, euc_dd]).T
        logger.info('clients idxes:{}'.format(client_idxes))
        # logger.info('metrics:{}'.format(tri_distance))
        logger.info('metrics sort:{}'.format(np.argsort(tri_distance, axis=0)))

        # compute covariance matrix and inverse matrix or pseudo-inverse matrix
        cov_matrix = np.cov(tri_distance.T)
        rank = np.linalg.matrix_rank(cov_matrix)
        if rank == cov_matrix.shape[0]:
            inv_matrix = np.linalg.inv(cov_matrix)
        else:
            inv_matrix = np.linalg.pinv(cov_matrix)

        # compute the mahalanobis distance
        ma_distances = []
        for i in range(num_clients):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        # take the p_num client idxes with the lowest score
        p_num = self.top_k * num_clients
        sorted_list = np.argpartition(scores, int(p_num))
        top_k_ind = sorted_list[:int(p_num)]
        other_ind = sorted_list[int(p_num):]

        # store the idxes and scores of benign clients and malicious clients
        benign_clients = [client_idxes[ti] for ti in top_k_ind]
        bgn_scores = [scores[i].round(2) for i in top_k_ind]
        adv_clients = [client_idxes[ti] for ti in other_ind]
        adv_scores = [scores[i].round(2) for i in other_ind]
        logger.info("The benign clients: {},\n\t scores:{}".format(benign_clients, bgn_scores))
        logger.info("The malicious clients: {},\n\t scores:{}".format(adv_clients, adv_scores))

        # aggregation
        selected_models = [client_models[i] for i in top_k_ind]
        selected_num_dps = [num_dps[i] for i in top_k_ind]
        global_model_state_dict = getattr(aggregator, aggregator_name)(selected_models, selected_num_dps)

        # return the aggregated global model state dict
        return global_model_state_dict
