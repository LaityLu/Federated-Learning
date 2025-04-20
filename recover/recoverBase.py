class RecoverBase:
    def __init__(self, dataset_train, dataset_test, dict_clients, list_num_dps, select_info, malicious_clients):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_clients = dict_clients
        self.list_num_dps = list_num_dps
        self.select_info = select_info
        self.malicious_clients = malicious_clients
        self.old_global_round = len(select_info)

    def remove_malicious_clients(self, c_ids, old_CM=None, *args, **kwargs):
        remaining_clients_id = []
        remaining_clients_models = []
        for i, client_id in enumerate(c_ids):
            if client_id not in self.malicious_clients:
                remaining_clients_id.append(client_id)
                if old_CM is not None:
                    remaining_clients_models.append(old_CM[i])
        num_dps = [self.list_num_dps[i] for i in remaining_clients_id]

        return remaining_clients_id, remaining_clients_models, num_dps
