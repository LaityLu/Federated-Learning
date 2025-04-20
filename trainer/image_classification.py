import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils import DatasetSplit


class ImageClassificationTrainer:
    def __init__(self,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 device
                 ):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.optimizer = optimizer
        self.device = device

    def update(self, dataset: Dataset, idxes, model: nn.Module, add_item=None, use_LBFGS=False):
        """
        :param dataset:
        :param idxes: the data points idxes
        :param model:
        :param add_item: the loss item ,such as FedProx
        :param use_LBFGS: it' for FedRecover
        :return:
        """
        model.to(self.device)
        # create the optimizer
        if use_LBFGS:
            optimizer = torch.optim.LBFGS(params=model.parameters(), lr=self.optimizer['args']['lr'], history_size=1,
                                          max_iter=4)
        else:
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
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_function(output, target)
                # it's designed for FedProx
                if add_item is not None:
                    loss += add_item(model)
                loss.backward()
                optimizer.step()
                # calculate the loss
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print the loss peer epoch
            # print(f'Epoch {epoch} Loss: {epoch_loss[-1]}')
        # return the updated model state dict and the average loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def eval(self, dataset: Dataset, model: nn.Module):
        """
        :param dataset:
        :param model:
        :return:
        """
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            batch_loss = []
            correct = 0
            data_size = len(dataset)
            test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_function(output, target)
                batch_loss.append(loss.item())
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            test_loss = sum(batch_loss) / len(batch_loss)
            test_accuracy = 100.00 * correct / data_size
            return test_accuracy, test_loss
