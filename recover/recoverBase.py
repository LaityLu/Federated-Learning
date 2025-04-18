class RecoverBase:
    def __init__(self, dataset_train, dataset_test, dict_clients, list_num_dps, select_info, malicious_clients):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_clients = dict_clients
        self.list_num_dps = list_num_dps
        self.select_info = select_info
        self.malicious_clients = malicious_clients
        self.old_global_round = len(select_info)

    def select_unlearned_clients(self, *args, **kwargs):
        pass
