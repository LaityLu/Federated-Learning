import copy
import numpy as np
import sklearn.metrics.pairwise as smp

from utils import parameters_dict_to_vector, setup_logger
logger = setup_logger()


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


def weighted_aggregation(global_model, client_models, weights, num_dps, lr):
    # weights = weights / np.sum(weights)
    freq = [snd / sum(num_dps) for snd in num_dps]
    agg_model = copy.deepcopy(global_model)
    for k in global_model.keys():
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        for i in range(len(client_models)):
            agg_model[k] += lr * (client_models[i][k] - global_model[k]) * freq[i] * weights[i]
    return agg_model


class FoolsGold:
    def __init__(self, use_memory=False, *args, **kwargs):
        self.memory_grads_dict = dict()
        self.use_memory = use_memory
        self.num_clients = 10
        self.FL_lr = 0.5  # for convergence

    def exec(self, global_model, client_models, client_idxes, num_dps, *args, **kwargs):
        logger.debug('FoolsGold begin::')

        self.num_clients = len(client_idxes)

        # flatten the model into a one-dimensional tensor
        v_global_model = parameters_dict_to_vector(global_model).detach().cpu().numpy()
        v_client_models = [parameters_dict_to_vector(cm).detach().cpu().numpy() for cm in client_models]

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
        weights = [w.round(3) for w in wv]
        logger.debug(f'foolsgold aggregation weights: {weights}')

        # aggregation
        global_model_state_dict = weighted_aggregation(global_model, client_models, wv, num_dps, self.FL_lr)

        logger.debug('FoolsGold end')

        # return the aggregated global model state dict
        return global_model_state_dict


