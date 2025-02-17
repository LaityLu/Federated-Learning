from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxes):
        self.dataset = dataset
        self.idxes = list(idxes)

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxes[item]]
        return image, label