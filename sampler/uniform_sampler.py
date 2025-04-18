import numpy as np


class UniformSampler:
    def __init__(self, dataset, num_clients, groupby=False, *args, **kwargs):
        """
        :param dataset:
        :param num_clients:
        :param groupby: ensure every client has all labels
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.groupby = groupby
        self.num_dps = len(dataset)
        self.num_dps_per_client = int(self.num_dps / self.num_clients)
        self.poison_images = kwargs.get('poison_images', None)

    def sample(self, *args, **kwargs):
        """
        :return: the dictionary of clients' data points idxes, such as
                    { 0:[213, 2423, 343], 1:[4432, 5123, 6432], ... 99:[4324, 3432, 1231] }
                 the num of data points per client, such as
                    [500, 500, 500, ...]
        """
        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [self.num_dps_per_client] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # if attack training, exclude the poisoning data points idxes
        if self.poison_images is not None:
            all_dps_idxes = list(set(all_dps_idxes) - set(self.poison_images['train']) - set(self.poison_images['test']))
        if not self.groupby:
            # just sample the idxes randomly
            for i in range(self.num_clients):
                dict_clients[i] = set(np.random.choice(all_dps_idxes, self.num_dps_per_client, replace=False))
                # prevent the data of the last client from being insufficient
                if len(all_dps_idxes) >= 2 * self.num_dps_per_client:
                    all_dps_idxes = list(set(all_dps_idxes) - dict_clients[i])
        else:
            if 'targets' not in dir(self.dataset):
                raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
            # get labels
            all_labels = self.dataset.targets
            labels = np.unique(all_labels)
            # if attack training, exclude the poisoning data points idxes
            if self.poison_images is not None:
                all_labels = np.delete(all_labels, list(set(self.poison_images['train']) | set(self.poison_images['test'])))
            all_labels = np.array(all_labels)
            all_dps_idxes = np.array(all_dps_idxes)
            # sample the idxes by labels
            for label in labels:
                label_idxes = all_dps_idxes[all_labels == label]
                np.random.shuffle(label_idxes)
                # uniformly assign
                for i in range(self.num_clients):
                    start_idx = i * self.num_dps_per_client // len(labels)
                    end_idx = (i + 1) * self.num_dps_per_client // len(labels)
                    temp_set = set(label_idxes[start_idx:end_idx])
                    if i not in dict_clients:
                        dict_clients[i] = set()
                    dict_clients[i] = dict_clients[i] | temp_set
        return dict_clients, list_num_dps