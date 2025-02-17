import copy
import numpy as np
import torch
import sklearn.metrics.pairwise as smp

from utils.logger_config import logger


# Takes in grad
# Compute similarity
# Get weightings
def fools_gold(grads, num_clients):
    """
    :param num_clients:
    :param grads: the gradients of clients
    :return: compute similarity and return weightings
    """

    cs = smp.cosine_similarity(grads) - np.eye(num_clients)

    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    # Per-row maximums
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Normalize learning rates to 0-1 range
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Element-wise logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # return the aggregation weights vector
    return wv


def weighted_aggregation(global_model, client_models, weights, num_dps):
    # freq = weights / np.sum(weights)
    freq = [snd / sum(num_dps) for snd in num_dps]
    # agg_model = copy.deepcopy(global_model)
    # for k in agg_model.keys():
    #     for i in range(len(client_models)):
    #         # it's to turn num_batches_tracked(int 64) to num_batches_tracked(float 32)
    #         agg_model[k] = agg_model[k].type(torch.float32)
    #         agg_model[k] += (client_models[i][k] - global_model[k]) * freq[i] * weights[i]
    # return agg_model
    agg_model = copy.deepcopy(client_models[0])
    for k in agg_model.keys():
        agg_model[k] = agg_model[k] * freq[0] * weights[0]
    for k in agg_model.keys():
        for i in range(1, len(client_models)):
            agg_model[k] += client_models[i][k] * freq[i] * weights[i]
    return agg_model


class FoolsGold:
    def __init__(self, use_memory=False):
        self.memory_grads_dict = dict()
        self.use_memory = use_memory
        self.num_clients = 10

    def exec(self, global_model, client_models, client_idxes, num_dps, *args, **kwargs):

        self.num_clients = len(client_idxes)

        # flatten the model into a one-dimensional tensor
        v_global_model = torch.cat([p.view(-1) for p in global_model.values()]).detach().cpu().numpy()
        v_client_models = [torch.cat([p.view(-1) for p in cm.values()]).detach().cpu().numpy() for cm in client_models]

        # compute the gradients
        grads = []
        memory_grads = []
        for i in range(self.num_clients):
            grads.append(v_client_models[i] - v_global_model)

        # store grads as history information
        for i in range(self.num_clients):
            if client_idxes[i] in self.memory_grads_dict.keys():
                self.memory_grads_dict[client_idxes[i]] += grads[i]
            else:
                self.memory_grads_dict[client_idxes[i]] = copy.deepcopy(grads[i])
            memory_grads.append(self.memory_grads_dict[client_idxes[i]])

        # turn to numpy array
        grads_array = np.array(grads)
        memory_grads_array = np.array(memory_grads)

        # use history information compute aggregation weight
        if self.use_memory:
            wv = fools_gold(memory_grads_array, self.num_clients)
        else:
            wv = fools_gold(grads_array, self.num_clients)
        logger.info(f'clients idxes:{client_idxes}')
        weights = [w.round(3) for w in wv]
        logger.info(f'foolsgold aggregation weights: {weights}')

        # aggregation
        global_model_state_dict = weighted_aggregation(global_model, client_models, wv, num_dps)

        # return the aggregated global model state dict
        return global_model_state_dict

# if __name__ == '__main__':
#     # test
#     import model
#
#     global_model = getattr(model, 'CNNCifar')(10).to(torch.device('cuda'))
#     model_1 = getattr(model, 'CNNCifar')(10).to(torch.device('cuda'))
#     model_2 = getattr(model, 'CNNCifar')(10).to(torch.device('cuda'))
#     client_models = [model_1.state_dict(), model_2.state_dict()]
#     client_idxes = [[11, 2], [3, 2]]
#     defense = FoolsGold(use_memory=True)
#     for i in range(2):
#         model = defense.exec(global_model.state_dict(), client_models, client_idxes[i])
