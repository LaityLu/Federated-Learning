import copy
import time

import aggregator
from utils import setup_logger
from .recoverBase import RecoverBase

logger = setup_logger()


class Retrain(RecoverBase):
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

    def recover(self, local_trainer, old_global_models, *args, **kwargs):
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(self.old_global_round):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, _, num_dps = self.remove_malicious_clients(self.select_info[rd])
            # begin training
            logger.info("----- Retrain Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []
            for i, idxes in enumerate(remaining_clients_id):
                local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                               copy.deepcopy(new_global_model))
                local_models.append(local_model)
                local_losses.append(local_loss)

            # aggregation
            global_model_state_dict = getattr(aggregator, self.aggregator)(local_models, num_dps)

            # update the global model
            new_global_model.load_state_dict(global_model_state_dict)

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
