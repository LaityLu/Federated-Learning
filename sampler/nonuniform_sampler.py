import numpy as np


class NonUniformSampler:
    # each client has at most 4 types of label
    def __init__(self, dataset, num_clients, is_attack=False, poison_images=None, **kwargs):
        """
        :param dataset:
        :param num_clients:
        :param args:
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_dps = len(dataset)
        self.num_dps_per_client = int(self.num_dps / self.num_clients)
        self.attack = is_attack
        if self.attack:
            self.poison_idxes = set(poison_images['train']) | set(poison_images['test'])

    def sample(self):
        """
        :return: the dictionary of clients' data points idxes, such as
                    { 0:[213, 2423, 343], 1:[4432, 5123, 6432], ... 99:[4324, 3432, 1231] }
                the num of data points per client, such as
                     [500, 500, 500, ...]
        """
        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [self.num_dps_per_client] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # get labels
        labels = self.dataset.targets
        # if attack training, exclude the poisoning data points idxes
        if self.attack:
            all_dps_idxes = list(set(all_dps_idxes) - self.poison_idxes)
            labels = np.delete(labels, list(self.poison_idxes))
        # sort all_dps_idxes by label
        idxes_labels = np.vstack((all_dps_idxes, labels))
        idxes_labels = idxes_labels[:, idxes_labels[1, :].argsort()]
        all_dps_idxes = idxes_labels[0, :]
        # divide and assign, each client has 2 shards data points
        num_shards = self.num_clients * 2
        idxes_shard = [i for i in range(num_shards)]
        for i in range(self.num_clients):
            rand_set = set(np.random.choice(idxes_shard, 2, replace=False))
            temp_set = set()
            # for each shard idx
            for rand in rand_set:
                temp_set = temp_set | set(
                    all_dps_idxes[rand * self.num_dps_per_client // 2:(rand + 1) * self.num_dps_per_client // 2])
            dict_clients[i] = temp_set
            # prevent the data of the last client from being insufficient
            if len(idxes_shard) >= 4:
                idxes_shard = list(set(idxes_shard) - rand_set)
        return dict_clients, list_num_dps