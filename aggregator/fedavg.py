import copy


def FedAvg(client_models_state_dict, num_dps=None, with_weight=False):
    """
    :param client_models_state_dict: a list of model state_dic
    :param num_dps: a list of the size of date for clients
    :param with_weight: aggregation weight based on data size
    :return: the averaged model state_dic
    """
    if with_weight:
        # compute aggregation weight based on data size
        weight = [snd / sum(num_dps) for snd in num_dps]
    else:
        weight = [1 / len(client_models_state_dict)] * len(client_models_state_dict)

    global_model = copy.deepcopy(client_models_state_dict[0])
    for k in global_model.keys():
        if k.split('.')[-1] != 'weight' and k.split('.')[-1] != 'bias':
            continue
        global_model[k] = global_model[k] * weight[0]
    for k in global_model.keys():
        if k.split('.')[-1] != 'weight' and k.split('.')[-1] != 'bias':
            continue
        for i in range(1, len(client_models_state_dict)):
            global_model[k] += client_models_state_dict[i][k] * weight[i]
    return global_model
