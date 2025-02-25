import torch
import numpy as np
from utils.logger_config import logger
from utils import parameters_dict_to_vector, visual_pca_whitening
import aggregator


class MultiMetrics:

    def __init__(self, top_k: float, adversary_list: list, *args, **kwargs):
        self.top_k = top_k
        self.adversary_list = adversary_list
        self.round = 0
        self.FNR = 0
        self.FPR = 0

    def exec(self, global_model, client_models, client_idxes, num_dps, aggregator_name, *args, **kwargs):
        # flatten the model into a one-dimensional tensor
        v_global_model = parameters_dict_to_vector(global_model).detach().cpu().numpy()
        v_client_models = [parameters_dict_to_vector(cm).detach().cpu().numpy() for cm in client_models]

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
        tri_distance0 = np.vstack([cos_dd, mht_dd, euc_dd]).T
        # visual_pca_whitening(tri_distance0, client_idxes, self.round)
        # self.round += 1
        tri_distance = tri_distance0 - tri_distance0.mean(axis=0)
        # tri_distance = tri_distance0
        # logger.info('metrics:{}'.format(tri_distance))
        # logger.info('metrics sort:{}'.format(np.argsort(tri_distance, axis=0)))

        # compute covariance matrix and inverse matrix or pseudo-inverse matrix
        cov_matrix = np.cov(tri_distance.T)
        # logger.info('covariance matrix:{}'.format(cov_matrix))

        rank = np.linalg.matrix_rank(cov_matrix)
        if rank == cov_matrix.shape[0]:
            inv_matrix = np.linalg.inv(cov_matrix)
        else:
            inv_matrix = np.linalg.pinv(cov_matrix)

        # whitening process
        w_distances = []
        for i in range(num_clients):
            t = tri_distance[i]
            w_dis = np.dot(np.dot(t, inv_matrix), t.T)
            w_distances.append(w_dis)

        scores = w_distances
        # take the p_num client idxes with the lowest score
        p_num = self.top_k * num_clients
        sorted_list = np.argpartition(scores, int(p_num))
        top_k_ind = sorted_list[:int(p_num)]
        other_ind = sorted_list[int(p_num):]

        # store the idxes and scores of benign clients and malicious clients
        benign_clients = [client_idxes[ti] for ti in top_k_ind]
        bgn_scores = [scores[i].round(2) for i in top_k_ind]
        bgn = np.vstack((benign_clients, bgn_scores))
        bgn = bgn[:, bgn[1, :].argsort()]
        adv_clients = [client_idxes[ti] for ti in other_ind]
        adv_scores = [scores[i].round(2) for i in other_ind]
        adv = np.vstack((adv_clients, adv_scores))
        adv = adv[:, adv[1, :].argsort()]
        logger.info("The benign clients: {},\n\t scores:{}".format(bgn[0], bgn[1]))
        logger.info("The malicious clients: {},\n\t scores:{}".format(adv[0], adv[1]))

        # aggregation
        selected_models = [client_models[i] for i in top_k_ind]
        selected_num_dps = [num_dps[i] for i in top_k_ind]
        global_model_state_dict = getattr(aggregator, aggregator_name)(selected_models, selected_num_dps)

        # return the aggregated global model state dict
        return global_model_state_dict
