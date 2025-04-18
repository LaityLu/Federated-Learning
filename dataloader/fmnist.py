from torchvision import datasets, transforms


def fmnist():
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True,
                                          transform=apply_transform)

    dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True,
                                         transform=apply_transform)
    return dataset_train, dataset_test
