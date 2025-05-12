import copy
import time

import numpy as np
import torch

import aggregator
from .fedEraser import FedEraser
import torch.nn.functional as F

from utils import setup_logger, model_to_traj

logger = setup_logger()


class Crab(FedEraser):
    def __init__(self,
                 dataset_train,
                 dataset_test,
                 dict_clients,
                 list_num_dps,
                 select_info,
                 malicious_clients,
                 *args,
                 **kwargs):
        super().__init__(dataset_train,
                         dataset_test,
                         dict_clients,
                         list_num_dps,
                         select_info,
                         malicious_clients)
        self.local_epochs = kwargs.get('local_epochs', 2)
        self.aggregator = kwargs.get('aggregator', 'FedAvg')
        self.P_rounds = kwargs.get('select_round_ratio', 0.6)
        self.X_clients = kwargs.get('select_client_ratio', 0.7)
        self.alpha = kwargs.get('alpha', 0.1)
        self.train_loss = kwargs.get('train_loss', None)
        self.window_size = 1
        self.list_select_rounds = []
        self.list_select_clients = []

    def recover(self, local_trainer, old_global_models, old_client_models, *args, **kwargs):
        start_time = time.time()
        start_round = 0
        start_loss = self.train_loss[0]
        for i in range(1, self.old_global_round):
            self.window_size += 1
            if self.train_loss[i] < start_loss * (1 - self.alpha) or i == self.old_global_round - 1:
                sl_round = self.select_round(start_round, old_global_models[start_round: i + 2])
                self.list_select_rounds.extend(sl_round)
                for rd in sl_round:
                    sel_clients_id = self.select_client_in_round(rd, old_global_models[rd + 1], old_client_models[rd])
                    self.list_select_clients.append(sel_clients_id)
                self.window_size = 0
                if i < self.old_global_round - 1:
                    start_round = i + 1
                    start_loss = self.train_loss[i + 1]
        logger.info(f'Crab select rounds: {self.list_select_rounds}')
        logger.info(f'Crab select clients: {self.list_select_clients}')

        rollback_round = self.adaptive_rollback()
        index = self.list_select_rounds.index(rollback_round)
        self.list_select_rounds = self.list_select_rounds[index:]
        self.list_select_clients = self.list_select_clients[index:]

        self.time_cost += time.time() - start_time

        sel_old_GM = []
        sel_old_CM = []
        for i, the_round in enumerate(self.list_select_rounds):
            sel_old_GM.append(old_global_models[the_round])
            old_CM_this_round = []
            for c_id in self.list_select_clients[i]:
                index = self.select_info[the_round].index(c_id)
                old_CM_this_round.append(old_client_models[the_round][index])
            sel_old_CM.append(old_CM_this_round)

        new_global_model_state = self.adaptive_recover(local_trainer, sel_old_GM, sel_old_CM)

        return new_global_model_state

    def select_round(self, start_epoch, old_global_models):
        rounds = [start_epoch + i for i in range(self.window_size)]
        logger.debug(f"The rounds in window: {rounds}")
        if self.window_size == 1:
            logger.debug(f'This time choose: {[start_epoch]}')
            return [start_epoch]
        k = int(self.window_size * self.P_rounds)
        GM_trajectory = model_to_traj(old_global_models)
        prior = GM_trajectory[0]
        kl_list = []
        for now_traj in GM_trajectory[1:]:
            kl = 0
            for module, prior_module in zip(now_traj, prior):
                log_x = F.log_softmax(module, dim=-1)
                y = F.softmax(prior_module, dim=-1)
                kl += F.kl_div(log_x, y, reduction='sum')
            kl_list.append(kl.cpu().item())
            prior = now_traj
        logger.debug(f"KL Divergence between global models in window:\n{kl_list}")
        kl_list = np.array(kl_list)
        sel_round = np.argsort(kl_list)[::-1]
        result = (sel_round[:k] + start_epoch).tolist()
        result.sort()
        logger.debug(f'This time choose: {result}')
        return result

    def select_client_in_round(self, rd, GM, CM_list):

        CM_list = model_to_traj(CM_list)
        k = int(len(CM_list) * self.X_clients)

        target_GM = [p.detach().clone() for p in GM.parameters()]

        similarity = []
        for client in CM_list:
            cos_sim = []
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    cos_sim.append(torch.mean(cos).cpu().item())
            similarity.append(np.mean(cos_sim))
        logger.debug(f'The old clients in this round: {self.select_info[rd]}')
        logger.debug(f'The similarity: {similarity}')
        sel_client = np.argsort(similarity)[::-1]
        sel_client = sel_client[:k].tolist()

        sel_client_id = [self.select_info[rd][idx] for idx in sel_client]

        logger.debug(f'Round {rd} choose: {sel_client_id}')

        return sel_client_id

    def adaptive_rollback(self):
        rollback_round = self.list_select_rounds[0]
        logger.info(f'Crab roll back to round: {rollback_round}')
        return rollback_round

    def adaptive_recover(self, local_trainer, old_global_models, old_client_models):
        local_trainer.local_epochs = self.local_epochs
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(len(self.list_select_rounds)):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models, num_dps = \
                self.remove_malicious_clients(self.list_select_clients[rd], old_client_models[rd])
            # begin training
            logger.info("----- Crab Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []

            if rd == 0:
                # the first recover round doesn't need calibration but aggregate the old cm directly
                old_cm_state = [remaining_clients_models[i].state_dict() for i in
                                range((len(remaining_clients_models)))]
                new_global_model.load_state_dict(getattr(aggregator, self.aggregator)(old_cm_state, num_dps))
                round_loss = 0
            else:
                for i, idxes in enumerate(remaining_clients_id):
                    local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                                   copy.deepcopy(new_global_model))
                    local_models.append(local_model)
                    local_losses.append(local_loss)

                # calibration and aggregation
                new_global_model.load_state_dict(self.calibration_training(old_global_models[rd],
                                                                           remaining_clients_models, new_global_model,
                                                                           local_models, num_dps))
                # compute the average loss in a round
                round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            round_losses.append(round_loss)

            self.time_cost += time.time() - start_time

            # testing
            # main accuracy
            test_accuracy, test_loss = local_trainer.eval(self.dataset_test, new_global_model)
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            MA.append(round(test_accuracy.item(), 2))

        logger.info("----- The recover process end -----")
        logger.info(f"Total time cost: {self.time_cost}s")
        logger.debug(f'Main Accuracy:{MA}')

        return new_global_model.state_dict()
