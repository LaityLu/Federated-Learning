import copy

import aggregator
from utils import setup_logger

logger = setup_logger()


class Retrain:
    def __init__(self, dataset_train, dataset_test, dict_clients, list_num_dps,
                 select_info, malicious_clients, *args, **kwargs):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_clients = dict_clients
        self.list_num_dps = list_num_dps
        self.select_info = select_info
        self.malicious_clients = malicious_clients
        self.old_global_round = len(select_info)
        self.aggregator = kwargs.get('aggregator', 'FedAvg')

    def recover(self, local_trainer, old_global_models, *args, **kwargs):
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(self.old_global_round):
            # select remaining clients
            remaining_clients_id = []
            for client_id in self.select_info[rd]:
                if client_id not in self.malicious_clients:
                    remaining_clients_id.append(client_id)
            num_dps = [self.list_num_dps[i] for i in remaining_clients_id]
            # begin training
            logger.info("----- Retrain Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            locals_losses = []
            local_models = []
            for i, idxes in enumerate(remaining_clients_id):
                local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                               copy.deepcopy(new_global_model))
                local_models.append(local_model)
                locals_losses.append(local_loss)

            # aggregation
            global_model_state_dict = getattr(aggregator, self.aggregator)(local_models, num_dps)

            # update the global model
            new_global_model.load_state_dict(global_model_state_dict)

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
