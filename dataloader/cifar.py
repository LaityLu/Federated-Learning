from torchvision import datasets, transforms


def cifar_dataset():
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True,
                                     transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True,
                                    transform=trans_cifar)
    return dataset_train, dataset_test