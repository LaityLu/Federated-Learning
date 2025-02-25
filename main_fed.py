import argparse
import copy
import random

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

import aggregator
import attack
import dataloader
import defense
import model
import sampler
import trainer
from utils.logger_config import logger, formatted_time

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/dba_multimetrics.yaml',
                        help='the path of config file')
    args = parser.parse_args()
    # load the config file
    try:
        with open(f'./{args.config}', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"can't find {args.config}")
    device = torch.device('cuda:{}'.format(config['gpu'])
                          if torch.cuda.is_available() and config['gpu'] != -1 else 'cpu')

    # get the dataset
    dataset_train, dataset_test = getattr(dataloader, config['Dataset']['name'])()

    # get the model
    global_model = getattr(model, config['Model']['name'])(**config['Model']['args'])
    global_model.to(device)

    # sampling data
    sampler = getattr(sampler, config['Sampler']['name'])(dataset_train,
                                                          **config['Sampler']['args'],
                                                          num_clients=config['FL']['num_clients'],
                                                          poison_images=config['Attack']['poison_images'])
    dict_clients, list_num_dps = sampler.sample()

    # get the trainer
    if config['FL']['is_attack']:
        # malicious trainer
        attacker = getattr(attack, config['Attack']['name'])(**config['Attack']['args'],
                                                             adversary_list=config['Attack']['adversary_list'],
                                                             poison_images=config['Attack']['poison_images'],
                                                             device=device)
        # normal trainer
        local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                    device=device)
    else:
        # normal trainer
        local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                    device=device)
    # get the defender
    if config['FL']['is_defense']:
        defender = getattr(defense, config['Defense']['name'])(**config['Defense']['args'])

    # fed training process
    num_select_clients = max(int(config['FL']['frac'] * config['FL']['num_clients']), 1)
    clients_idxes = np.arange(config['FL']['num_clients'])
    if config['FL']['is_attack']:
        benign_client_idxes = np.setdiff1d(clients_idxes, config['Attack']['adversary_list'])
    else:
        benign_client_idxes = clients_idxes
    round_losses = []
    for rd in range(config['FL']['round']):
        # select clients and store their dataset size
        select_clients = np.random.choice(benign_client_idxes, num_select_clients, replace=False).tolist()
        if config['FL']['is_attack'] and rd in config['Attack']['attack_round']:
            # replace the benign clients with malicious clients
            if config['Attack']['name'] == 'SemanticAttack':
                select_adv = config['Attack']['adversary_list']
            else:
                # DBA
                select_adv = random.sample(config['Attack']['adversary_list'], 2)
            for i, adv_idxes in enumerate(select_adv):
                select_clients[i] = adv_idxes
        num_dps = [list_num_dps[i] for i in select_clients]
        # begin training
        print("-----  Round {:3d}  -----".format(rd))
        logger.info("Round {:3d}:".format(rd))
        logger.info('selected clients:{}'.format(select_clients))
        # store the local loss and local model for each client
        locals_losses = []
        local_models = []
        for i, idxes in enumerate(select_clients):
            # confrontation model training
            if config['FL']['is_attack'] and rd in config['Attack']['attack_round'] and idxes in config['Attack']['adversary_list']:
                logger.info('malicious client:{}'.format(idxes))
                local_model, local_loss = attacker.exec(dataset_train, dict_clients[idxes],
                                                        copy.deepcopy(global_model), adversarial_index=idxes)
                have_attack = True
            # normal model training
            else:
                local_model, local_loss = local_trainer.update(dataset_train, dict_clients[idxes],
                                                               copy.deepcopy(global_model))
            local_models.append(local_model)
            locals_losses.append(local_loss)
        # defense and aggregation
        if config['FL']['is_defense']:
            global_model_state_dict = defender.exec(global_model.state_dict(), local_models,
                                                    select_clients, num_dps, config['FL']['aggregator'])
        else:
            # normal aggregation
            global_model_state_dict = getattr(aggregator, config['FL']['aggregator'])(local_models, num_dps)

        # update global model
        global_model.load_state_dict(global_model_state_dict)

        # compute the average loss in a round
        round_loss = sum(locals_losses) / len(locals_losses)
        print('Training average loss {:.3f}'.format(round_loss))
        logger.info('Training average loss {:.3f}'.format(round_loss))
        round_losses.append(round_loss)

        # testing
        # train_accuracy, train_loss = local_trainer.eval(dataset_train, global_model)
        # print("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))
        # logger.info("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))
        # main accuracy
        test_accuracy, test_loss = local_trainer.eval(dataset_test, global_model)
        print("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
        logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
        # backdoor accuracy
        if config['FL']['is_attack']:
            # and rd >= config['Attack']['attack_round'][0]
            if config['Attack']['name'] == 'SemanticAttack':
                test_attack_accuracy, _ = attacker.eval(dataset_train, global_model)
            else:
                # DBA
                test_attack_accuracy, _ = attacker.eval(dataset_test, global_model)
            print("Attack accuracy: {:.2f}%".format(test_attack_accuracy))
            logger.info("Attack accuracy: {:.2f}%".format(test_attack_accuracy))

    # save the final global model
    if config['FL']['save_model']:
        torch.save(global_model.state_dict(), './save/save_model/{}_{}.pth'.format(config['Model']['name'], formatted_time))

    # plot training loss curve
    if config['FL']['plot_loss_curve']:
        plt.figure()
        plt.plot(range(len(round_losses)), round_losses)
        plt.ylabel('train_loss')
        plt.savefig('./save/loss/{}.png'.format(formatted_time))
