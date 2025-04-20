import copy

import torch

import aggregator
from .recoverBase import RecoverBase
from utils import setup_logger

logger = setup_logger()


class FedEraser(RecoverBase):
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
        self.round_interval = kwargs.get('round_interval', 1)
        self.local_epochs = kwargs.get('local_epochs', 2)
        self.aggregator = kwargs.get('aggregator', 'FedAvg')

    def recover(self, local_trainer, old_global_models, old_client_models, *args, **kwargs):
        local_trainer.local_epochs = self.local_epochs
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(0, self.old_global_round, self.round_interval):
            # select remaining clients
            remaining_clients_id, remaining_clients_models, num_dps = \
                self.remove_malicious_clients(self.select_info[rd], old_client_models[rd])
            # begin training
            logger.info("----- FedEraser Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            locals_losses = []
            local_models = []

            if rd == 0:
                # the first recover round doesn't need calibration but aggregate the old client models directly
                old_cm_state = [remaining_clients_models[i].state_dict() for i in range((len(remaining_clients_models)))]
                new_global_model.load_state_dict(getattr(aggregator, self.aggregator)(old_cm_state, num_dps))
                round_loss = 0
            else:
                for i, idxes in enumerate(remaining_clients_id):
                    local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                                   copy.deepcopy(new_global_model))
                    local_models.append(local_model)
                    locals_losses.append(local_loss)
                # calibration and aggregation
                new_global_model.load_state_dict(self.calibration_training(old_global_models[rd],
                                                                           remaining_clients_models, new_global_model,
                                                                           local_models, num_dps))
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

    def calibration_training(self, old_gm, old_cm, new_gm, new_cm_state, num_dps):

        new_global_model_state = new_gm.state_dict()  # newGM_t
        old_global_model_state = old_gm.state_dict()  # oldGM_t

        return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

        assert len(old_cm) == len(new_cm_state)

        old_cm_state = [old_cm[i].state_dict() for i in range((len(old_cm)))]

        # aggregation
        old_param_update = getattr(aggregator, self.aggregator)(old_cm_state, num_dps)
        new_param_update = getattr(aggregator, self.aggregator)(new_cm_state, num_dps)

        for layer in old_global_model_state.keys():
            return_model_state[layer] = old_global_model_state[layer]
            if layer.split('.')[-1] == 'num_batches_tracked':
                continue
            old_param_update[layer] = old_param_update[layer] - old_global_model_state[layer]  # oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - new_global_model_state[layer]  # newCM - newGM_t

            step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
            step_direction = new_param_update[layer] / torch.norm(
                new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

            return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

        return return_model_state
