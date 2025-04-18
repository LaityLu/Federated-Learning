import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from utils import DatasetSplit, setup_logger

logger = setup_logger()


def get_dis_loss(g_model, l_model):
    # flatten the model into a one-dimensional tensor
    v_g = torch.cat([p.view(-1) for p in g_model.values()]).detach().cpu().numpy()
    v_l = torch.cat([p.view(-1) for p in l_model.values()]).detach().cpu().numpy()
    distance = float(
        np.linalg.norm(v_g - v_l))
    return distance


def add_pixel_pattern(origin_image, adversarial_index, trigger_args):
    image = copy.deepcopy(origin_image)
    # triggers' params
    poison_patterns = []

    # add global trigger
    if adversarial_index == -1:
        for i in range(0, trigger_args['trigger_num']):
            poison_patterns = poison_patterns + trigger_args[str(i) + '_poison_pattern']
    else:
        # add local trigger
        poison_patterns = trigger_args[str(adversarial_index) + '_poison_pattern']
    if trigger_args['channels'] == 3:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
    elif trigger_args['channels'] == 1:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1

    # return the image with trigger
    return image


def get_test_data_idxes(dataset, remove_label: int):
    # delete the test data with poisoning label
    if 'targets' not in dir(dataset):
        raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
    test_data_idxes = np.array([], dtype=int)
    all_dps_idxes = np.arange(len(dataset), dtype=int)
    # get labels
    all_labels = dataset.targets
    labels = np.unique(all_labels)
    # sample the idxes by labels
    for label in labels:
        if label == remove_label:
            continue
        label_idxes = all_dps_idxes[all_labels == label]
        test_data_idxes = np.concatenate((test_data_idxes, label_idxes))
    np.random.shuffle(test_data_idxes)

    return test_data_idxes


class DBA:
    def __init__(self,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 poisoning_per_batch: int,
                 stealth_rate: int,
                 poison_label_swap: int,
                 trigger: dict,
                 adversary_list: list,
                 device,
                 **kwargs
                 ):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.optimizer = optimizer
        self.device = device
        self.trigger_args = trigger
        self.poisoning_per_batch = poisoning_per_batch
        self.stealth_rate = stealth_rate
        self.poison_label_swap = poison_label_swap
        self.adversary_list = adversary_list
        self.test_data_idxes = None

    def exec(self, dataset: Dataset, data_idxes, initial_model: nn.Module, adversarial_index):
        adversarial_index = self.adversary_list.index(adversarial_index) % 4

        # copy the global model in the last round
        global_model = dict()
        for key, value in initial_model.state_dict().items():
            global_model[key] = value.clone().detach().requires_grad_(False)

        initial_model.to(self.device)

        # create the optimizer
        optimizer = getattr(optim, self.optimizer['name'])(initial_model.parameters(), **self.optimizer['args'])

        # load dataset
        train_loader = DataLoader(DatasetSplit(dataset, data_idxes), batch_size=self.batch_size, shuffle=True)

        # start training
        initial_model.train()
        # store the loss peer epoch
        epoch_loss = []
        for epoch in range(self.local_epochs):
            # store the loss for each batch
            batch_loss = []
            # poisoning data for each batch
            for batch_idx, (images, labels) in enumerate(train_loader):
                for i in range(self.poisoning_per_batch):
                    if i == len(images):
                        break
                    images[i] = add_pixel_pattern(images[i], adversarial_index, self.trigger_args)
                    labels[i] = self.poison_label_swap
                images = images.to(self.device)
                labels = labels.to(self.device)
                initial_model.zero_grad()
                output = initial_model(images)
                # compute classification loss
                class_loss = self.loss_function(output, labels)
                # compute distance loss
                distance_loss = get_dis_loss(global_model, initial_model.state_dict())
                # compute the final loss
                loss = (1 - self.stealth_rate) * class_loss + self.stealth_rate * distance_loss
                loss.backward()
                optimizer.step()
                # calculate the loss
                batch_loss.append(loss.item())
                # print(sum(batch_loss) / len(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # return the updated model state dict and the average loss
            return initial_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def eval(self, dataset: Dataset, eval_model: nn.Module):
        if self.test_data_idxes is not None:
            pass
        else:
            self.test_data_idxes = get_test_data_idxes(dataset, self.poison_label_swap)
        test_data_idxes = self.test_data_idxes
        eval_model = eval_model.to(self.device)
        eval_model.eval()
        with torch.no_grad():
            # store the loss and num of correct classification
            batch_loss = []
            correct = 0
            # load poisoning test data
            data_size = len(dataset)
            ldr_test = DataLoader(DatasetSplit(dataset, idxes=test_data_idxes),
                                  batch_size=self.batch_size)
            for batch_idx, (images, labels) in enumerate(ldr_test):
                for i in range(len(images)):
                    images[i] = add_pixel_pattern(images[i], -1, self.trigger_args)
                    labels[i] = self.poison_label_swap
                images, labels = images.to(self.device), labels.to(self.device)
                output = eval_model(images)
                labels.fill_(self.poison_label_swap)
                # compute the loss
                loss = self.loss_function(output, labels)
                batch_loss.append(loss.item())
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            test_loss = sum(batch_loss) / len(batch_loss)
            test_accuracy = 100.00 * correct / data_size

        return test_accuracy, test_loss


if __name__ == '__main__':
    import model
    from torchvision import datasets, transforms
    import random

    config = {'args':
                  {'local_epochs': 100,
                   'batch_size': 64,
                   'loss_function': 'nll_loss',
                   'optimizer':
                       {'name': 'SGD',
                        'args':
                            {'lr': 0.01,
                             'momentum': 0.9}},
                   'poisoning_per_batch': 10,
                   'poison_label_swap': 2,
                   'stealth_rate': 0,
                   'trigger':
                       {'channels': 3,
                        'trigger_num': 4,
                        '0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
                        '1_poison_pattern': [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
                        '2_poison_pattern': [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
                        '3_poison_pattern': [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
                        }
                   }
              }
    attack = DBA(**config['args'], device=torch.device('cuda'))
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = datasets.CIFAR10('../../data/cifar', train=True, download=True,
                                     transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../../data/cifar', train=False, download=True,
                                    transform=trans_cifar)
    global_model = getattr(model, 'CNNCifar')(10).to(torch.device('cuda'))
    train_idxes = [random.randint(0, 49999) for _ in range(500)]
    test_idxes = [random.randint(0, 9999) for _ in range(500)]
    model, _ = attack.exec(dataset_train, train_idxes, global_model, 0)
    global_model.load_state_dict(model)
    test_acc, _ = attack.eval(dataset_test, global_model)
