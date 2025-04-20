import copy

import aggregator
from .recoverBase import RecoverBase
from utils import setup_logger

logger = setup_logger()


class FedRecover(RecoverBase):
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
        self.aggregator = kwargs.get('aggregator', 'FedAvg')
        self.T_w = kwargs.get('warm_up_rounds', 10)
        self.T_c = kwargs.get('correction_period', 10)
        self.alpha = kwargs.get('alpha', 0.000001)
        self.T_f = kwargs.get('final_tuning_rounds', 5)
        self.buffer_size = kwargs.get('buffer_size', 1)
        self.prev_train_loss = 10

    def recover(self, local_trainer, old_global_models, old_client_models, *args, **kwargs):
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(self.old_global_round):
            # select remaining clients
            remaining_clients_id, remaining_clients_models, num_dps = \
                self.remove_malicious_clients(self.select_info[rd], old_client_models[rd])
            # begin training
            logger.info("----- FedRecover Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            locals_losses = []
            local_models = []

            for i, idxes in enumerate(remaining_clients_id):
                local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                               copy.deepcopy(new_global_model))
                local_models.append(local_model)
                locals_losses.append(local_loss)

            # compute the average loss in a round
            round_loss = sum(locals_losses) / len(locals_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            round_losses.append(round_loss)

            # testing
            # main accuracy
            test_accuracy, test_loss = local_trainer.eval(self.dataset_test, new_global_model)
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            MA.append(round(test_accuracy.item(), 2))

        logger.info("----- The recover process end -----")
        logger.debug(f'Main Accuracy:{MA}')

        return new_global_model.state_dict()
