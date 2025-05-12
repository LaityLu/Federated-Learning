import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils import DatasetSplit, setup_logger
from opacus import PrivacyEngine

logger = setup_logger()


class ImageClassificationTrainer:
    def __init__(self,
                 client_id,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 device,
                 **kwargs
                 ):
        self.id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.optimizer = optimizer
        self.device = device
        self.with_DP = kwargs.get('with_DP', True)
        self.train_data = None
        if self.with_DP:
            self.noise_multiplier = kwargs.get('noise_multiplier', 1.3)
            self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
            self.delta = kwargs.get('delta', 1e-5)
            self.privacy_engine = PrivacyEngine()

    def update(self, dataset: Dataset, idxes, model: nn.Module):
        """
        :param dataset:
        :param idxes: the data points idxes
        :param model:
        :return:
        """
        if self.train_data is None:
            self.train_data = DatasetSplit(dataset, idxes)

        model.to(self.device)
        # create the optimizer
        optimizer = getattr(optim, self.optimizer['name'])(model.parameters(), **self.optimizer['args'])

        # load dataset
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        # start training
        model.train()

        model, optimizer, train_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            # module=model,
            # optimizer=optimizer,
            # data_loader=train_loader,
            # target_delta=1e-5,
            # target_epsilon=5,
            # epochs=10,
            # max_grad_norm=1.0,
        )

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
                loss.backward()
                optimizer.step()
                # calculate the loss
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print the loss peer epoch
            # print(f'Epoch {epoch} Loss: {epoch_loss[-1]}')
            if self.with_DP:
                epsilon, best_alpha = self.privacy_engine.accountant.get_epsilon(self.delta, [5.0])
                logger.info(
                    f"Client: {self.id}\t"
                    f"(ε = {epsilon:.2f}, δ = {self.delta}, alpha = {best_alpha})")

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
