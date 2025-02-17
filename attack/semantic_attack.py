import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils import DatasetSplit
# from utils.logger_config import logger


def get_dis_loss(g_model, l_model):
    # flatten the model into a one-dimensional tensor
    v_g = torch.cat([p.view(-1) for p in g_model.values()]).detach().cpu().numpy()
    v_l = torch.cat([p.view(-1) for p in l_model.values()]).detach().cpu().numpy()
    distance = float(
        np.linalg.norm(v_g - v_l))
    return distance


class SemanticAttack:
    def __init__(self,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 stealth_rate: float,
                 scale_weight: float,
                 poison_images: dict,
                 device
                 ):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.optimizer = optimizer
        self.poison_images = poison_images
        self.stealth_rate = stealth_rate
        self.scale_weight = scale_weight
        self.device = device

    def exec(self, dataset: Dataset, idxes, model: nn.Module):
        """
        :param dataset:
        :param idxes: idxes: the data points idxes
        :param model:
        :param logger:
        :return:
        """
        # copy the global model in the last round
        global_model = dict()
        for key, value in model.state_dict().items():
            global_model[key] = value.clone().detach().requires_grad_(False)

        model.to(self.device)

        # create the optimizer
        optimizer = getattr(optim, self.optimizer['name'])(model.parameters(), **self.optimizer['args'])

        # load dataset
        train_loader = DataLoader(DatasetSplit(dataset, idxes), batch_size=self.batch_size, shuffle=True)

        # start training
        model.train()
        # store the loss peer epoch
        epoch_loss = []
        for epoch in range(self.local_epochs):
            # store the loss for each batch
            batch_loss = []
            # poisoning data for each batch
            for batch_idx, (images, labels) in enumerate(train_loader):
                for i in range(len(self.poison_images['train'])):
                    images[i] = dataset[self.poison_images['train'][i]][0]
                    # add gaussian noise
                    images[i].add_(torch.FloatTensor(images[i].shape).normal_(0, 0.01))
                    labels[i] = self.poison_images['target_label']
                images = images.to(self.device)
                labels = labels.to(self.device)
                model.zero_grad()
                output = model(images)
                # compute classification loss
                class_loss = self.loss_function(output, labels)
                # compute distance loss
                distance_loss = get_dis_loss(global_model, model.state_dict())
                # compute the final loss
                loss = (1 - self.stealth_rate) * class_loss + self.stealth_rate * distance_loss
                loss.backward()
                optimizer.step()
                # calculate the loss
                batch_loss.append(loss.item())
                # print(sum(batch_loss) / len(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print the loss peer epoch
            # print(f'Epoch {epoch} Loss: {epoch_loss[-1]}')
            # # 测试对抗性模型在投毒数据上的准确率和损失
            # acc_p, loss_p = self.attack_test(model, dataset)
            # print(loss_p)
            # # 更新学习率
            # if loss_p <= 0.0001:
            #     if self.step_lr:
            #         scheduler.step()
            #         print("step_lr")
        # test the acc and loss of poisoning model on poisoning data
        acc_p, loss_p = self.eval(dataset, model)
        print("local model attack accuracy:{:.2f}%, loss:{:.4f}".format(acc_p, loss_p))
        # logger.info("local model attack accuracy:{:.2f}%, loss:{:.4f}".format(acc_p, loss_p))

        # test the acc and loss of poisoning model on clean test data
        # acc_train, loss_train = test_img(model, dataset, self.args)
        # print("Training accuracy: {:.2f}%, loss: {:.3f}".format(acc_train, loss_train))
        # logger.info("Training accuracy: {:.2f}%, loss: {:.3f}".format(acc_train, loss_train))
        # scale the model weight
        for key, value in model.state_dict().items():
            new_value = global_model[key] + (value - global_model[key]) * self.scale_weight
            model.state_dict()[key].copy_(new_value)

        # return the updated model state dict and the average loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def eval(self, dataset: Dataset, model: nn.Module):
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            # store the loss and num of correct classification
            batch_loss = []
            correct = 0
            # load poisoning test data
            data_size = len(self.poison_images['test'])
            ldr_test = DataLoader(DatasetSplit(dataset, idxes=self.poison_images['test']),
                                  batch_size=data_size)
            for batch_idx, (data, target) in enumerate(ldr_test):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                target.fill_(self.poison_images['target_label'])
                # compute the loss
                loss = self.loss_function(output, target)
                batch_loss.append(loss.item())
                y_pred = output.data.max(1, keepdim=True)[1]
                print("attack pred:{}".format(y_pred.tolist()))
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            test_loss = sum(batch_loss) / len(batch_loss)
            test_accuracy = 100.00 * correct / data_size

        return test_accuracy, test_loss


if __name__ == '__main__':
    import model
    from torchvision import datasets, transforms
    import random

    config = {'attack_round': [4],
              'poison_images':
                  {'target_label': 2,
                   'train': [2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744,
                             14209, 14238, 18716, 19793, 20781],
                   'test': [21529, 31311, 40518, 40633, 42119, 42663, 49392]},
              'args':
                  {'local_epochs': 100,
                   'batch_size': 64,
                   'loss_function': 'nll_loss',
                   'optimizer':
                       {'name': 'SGD',
                        'args':
                            {'lr': 0.01,
                             'momentum': 0.8}},
                   'stealth_rate': 0,
                   'scale_weight': 10
                   }
              }
    attack = SemanticAttack(**config['args'], poison_images=config['poison_images'], device=torch.device('cuda'))
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = datasets.CIFAR10('../../data/cifar', train=True, download=True,
                                     transform=trans_cifar)
    global_model = getattr(model, 'CNNCifar')(10).to(torch.device('cuda'))
    idxes = [random.randint(0, 49999) for _ in range(500)]
    model, _ = attack.exec(dataset_train, idxes, global_model)
