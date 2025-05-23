import numpy as np


class DirichletSampler:
    def __init__(self, dataset, num_clients, alpha, *args, **kwargs):
        """
        :param dataset:
        :param num_clients:
        :param alpha: concentration parameters of Dirichlet distribution
        """
        self.dataset = dataset
        self.num_dps = len(dataset)
        self.num_clients = num_clients
        self.alpha = alpha
        self.poison_images = kwargs.get('poison_images', None)

    def sample(self, *args, **kwargs):
        """
        :return: the dictionary of clients' data points idxes, such as
                    { 0:[213, 2423, 343], 1:[4432, 5123, 6432], ... 99:[4324, 3432, 1231] }
                 the num of data points per client, such as
                    [500, 500, 500, ...]
        """
        # judge whether the attribute 'targets' is in the dataset
        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [0] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # get labels
        all_labels = self.dataset.targets
        labels = np.unique(all_labels)
        # if attack training requires it, exclude the poisoning data points idxes
        if self.poison_images is not None:
            all_dps_idxes = list(set(all_dps_idxes) - set(self.poison_images['train']) - set(self.poison_images['test']))
            all_labels = np.delete(all_labels, list(set(self.poison_images['train']) | set(self.poison_images['test'])))
        all_labels = np.array(all_labels)
        all_dps_idxes = np.array(all_dps_idxes)
        # produce the category proportion of each client
        proportions = np.random.dirichlet([self.alpha] * len(labels), self.num_clients)
        # for each label
        for c in labels:
            label_idxes = all_dps_idxes[all_labels == c]
            np.random.shuffle(label_idxes)
            proportions_c = proportions[:, c]
            proportions_c = (proportions_c / proportions_c.sum()) * len(label_idxes)
            proportions_c = proportions_c.astype(int)
            proportions_c[-1] = len(label_idxes) - proportions_c[:-1].sum()

            split_label_idxes = np.split(label_idxes, np.cumsum(proportions_c)[:-1])

            for client_idx, idxes in enumerate(split_label_idxes):
                if client_idx not in dict_clients:
                    dict_clients[client_idx] = []
                dict_clients[client_idx].extend(idxes)
        # compute data points num of each client
        for i in range(self.num_clients):
            list_num_dps[i] = len(dict_clients[i])

        return dict_clients, list_num_dps
