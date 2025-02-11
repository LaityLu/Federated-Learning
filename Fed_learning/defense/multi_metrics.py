import torch
import numpy as np
from utils.logger_config import logger


class MultiMetrics:

    def __init__(self, top_k: float):
        self.top_k = top_k

    def exec(self, global_model, client_models, client_idxes):
        # flatten the model into a one-dimensional tensor
        v_global_model = torch.cat([p.view(-1) for p in global_model.values()]).detach().cpu().numpy()
        v_client_models = [torch.cat([p.view(-1) for p in cm.values()]).detach().cpu().numpy() for cm in client_models]

        # store the distance between each client model and global model
        cos_dis = [0.0] * len(v_client_models)  # cosine distance
        euc_dis = [0.0] * len(v_client_models)  # euclidean distance
        mht_dis = [0.0] * len(v_client_models)  # manhattan distance
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
        cos_dd = [0.0] * len(v_client_models)
        euc_dd = [0.0] * len(v_client_models)
        mht_dd = [0.0] * len(v_client_models)
        for i in range(len(v_client_models)):
            for j in range(len(v_client_models)):
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
        logger.info('metrics:{}'.format(tri_distance))
        logger.info('metrics sort:{}'.format(np.argsort(tri_distance, axis=0)))

        # compute covariance matrix and inverse matrix
        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        # compute the mahalanobis distance
        ma_distances = []
        for i in range(len(v_client_models)):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        # take the p_num client idxes with the lowest score
        p_num = self.top_k * len(scores)
        top_k_ind = np.argpartition(scores, int(p_num))[:int(p_num)]
        # store the idxes and scores of benign clients and malicious clients
        benign_users = [client_idxes[ti] for ti in top_k_ind]
        bgn_scores = [scores[i] for i in top_k_ind]
        adv_users = list(set(client_idxes) - set(benign_users))
        adv_scores = list(set(scores) - set(bgn_scores))
        bgn_scores = [score.round(2) for score in bgn_scores]
        adv_scores = [score.round(2) for score in adv_scores]
        logger.info("The benign users: {},\n\t scores:{}".format(benign_users, bgn_scores))
        logger.info("The malicious users: {},\n\t scores:{}".format(adv_users, adv_scores))

        # return the idxes of benign clients and malicious clients
        return benign_users, adv_users
