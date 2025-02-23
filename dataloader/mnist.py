from torchvision import datasets, transforms


def mnist_dataset():
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                                  transform=trans_mnist)
    return dataset_train, dataset_test
