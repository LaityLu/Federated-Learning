import copy


def FedAvg(client_models, num_dps):
    """
    :param client_models: a list of model state_dic
    :param num_dps: a list of the size of date for clients
    :return: the averaged model state_dic
    """
    # compute aggregation weight based on data size
    freq = [snd / sum(num_dps) for snd in num_dps]

    global_model = copy.deepcopy(client_models[0])
    for k in global_model.keys():
        global_model[k] = global_model[k] * freq[0]
    for k in global_model.keys():
        for i in range(1, len(client_models)):
            global_model[k] += client_models[i][k] * freq[i]
    return global_model
