import argparse
import copy

import os.path
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

from utils import setup_logger, check, save_clients_data, save_global_model, save_client_model, \
    save_select_info, load_clients_data, save_train_loss

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dba_ours',
                        help='the path of config file')
    args = parser.parse_args()
    # load the config file
    try:
        with open(f'./config/{args.config}.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"can't find {args.config}")
    device = torch.device('cuda:{}'.format(config['gpu'])
                          if torch.cuda.is_available() and config['gpu'] != -1 else 'cpu')

    # set the logger
    log_file_path = os.path.join('./save/logs', config['Dataset']['name'])
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = setup_logger(log_file_path + f'/{args.config}.log')

    # get the dataset
    dataset_train, dataset_test = getattr(dataloader, config['Dataset']['name'])()

    # get the model
    global_model = getattr(model, config['Model']['name'])(**config['Model']['args'])
    global_model.to(device)
    # save the initial global model
    if config['FL']['is_recover']:
        save_global_model(config['Dataset']['name'], 0, global_model)

    # sampling data
    # set the dir path for saving client data
    clients_data_dir_path = os.path.join('./save/data_sampled', config['Dataset']['name'])
    if not os.path.exists(clients_data_dir_path):
        os.makedirs(clients_data_dir_path)
    # check whether that data has been sampled
    if check(clients_data_dir_path + '/data_config.json', config):
        # load client data
        dict_clients, list_num_dps = load_clients_data(clients_data_dir_path)
    else:
        # sampling data to clients
        sampler = getattr(sampler, config['Sampler']['name'])(dataset_train,
                                                              **config['Sampler']['args'],
                                                              num_clients=config['FL']['num_clients'])
        dict_clients, list_num_dps = sampler.sample()
        # save client data
        save_clients_data(clients_data_dir_path, config, dict_clients, list_num_dps)

    # get the trainer
    # normal trainer
    local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                device=device)
    if config['FL']['is_attack']:
        # malicious trainer
        attacker = getattr(attack, config['Attack']['name'])(**config['Attack']['args'],
                                                             adversary_list=config['Attack']['adversary_list'],
                                                             device=device)

    # get the defender
    if config['FL']['is_defense']:
        defender = getattr(defense, config['Defense']['name'])(**config['Defense']['args'],
                                                               adversary_list=config['Attack']['adversary_list'])

    # fed training process
    select_info = []
    num_select_clients = max(int(config['FL']['frac'] * config['FL']['num_clients']), 1)
    clients_idxes = np.arange(config['FL']['num_clients'])
    if config['FL']['is_attack']:
        benign_client_idxes = np.setdiff1d(clients_idxes, config['Attack']['adversary_list'], assume_unique=True)
    else:
        benign_client_idxes = clients_idxes
    # malicious_records = [0] * config['FL']['num_clients']
    round_losses = []
    MA = []
    BA = []
    for rd in range(config['FL']['round']):
        # select clients and store their dataset size
        if not config['FL']['is_attack'] or config['Attack']['random_attack'] \
                or rd not in config['Attack']['attack_round']:
            # no attack or random attack or the attacker behave benign in this round
            select_clients = np.random.choice(clients_idxes, num_select_clients, replace=False).tolist()
        else:
            # fixed attacker attack in  fixed rounds
            select_clients = np.random.choice(benign_client_idxes, num_select_clients - \
                                              config['Attack']['num_adv_each_round'], replace=False).tolist()
            select_adv = random.sample(config['Attack']['adversary_list'], config['Attack']['num_adv_each_round'])
            select_clients += select_adv
        select_info.append(select_clients)
        num_dps = [list_num_dps[i] for i in select_clients]

        # begin training
        logger.info("-----  Round {:3d}  -----".format(rd))
        logger.info('selected clients:{}'.format(select_clients))
        # store the local loss and local model for each client
        locals_losses = []
        local_models = []
        for i, idxes in enumerate(select_clients):
            if config['FL']['is_attack'] and idxes in config['Attack']['adversary_list'] and (
                    config['Attack']['random_attack'] or rd in config['Attack']['attack_round']):
                # poisoning model training
                logger.info('malicious client {} attacked'.format(idxes))
                local_model, local_loss = attacker.exec(dataset_train, dict_clients[idxes],
                                                        copy.deepcopy(global_model), adversarial_index=idxes)
            # normal model training
            else:
                local_model, local_loss = local_trainer.update(dataset_train, dict_clients[idxes],
                                                               copy.deepcopy(global_model))
            local_models.append(local_model)
            locals_losses.append(local_loss)
            # save the client models
            if config['FL']['is_recover']:
                save_client_model(config['Dataset']['name'], rd, local_models)

        # defense and aggregation
        if config['FL']['is_defense']:
            global_model_state_dict, mal_clients = defender.exec(global_model.state_dict(), local_models,
                                                                 select_clients, num_dps, config['FL']['aggregator'])
            # for idx in mal_clients:
            #     malicious_records[idx] += 1
        else:
            # normal aggregation
            global_model_state_dict = getattr(aggregator, config['FL']['aggregator'])(local_models, num_dps)

        # update global model
        global_model.load_state_dict(global_model_state_dict)
        # save the global models of each round
        if config['FL']['is_recover']:
            save_global_model(config['Dataset']['name'], rd + 1, global_model)

        # compute the average loss in a round
        round_loss = sum(locals_losses) / len(locals_losses)
        logger.info('Training average loss: {:.3f}'.format(round_loss))
        round_losses.append(round_loss)

        # testing
        # train_accuracy, train_loss = local_trainer.eval(dataset_train, global_model)
        # print("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))
        # logger.info("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))
        # main accuracy
        test_accuracy, test_loss = local_trainer.eval(dataset_test, global_model)
        logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
        MA.append(round(test_accuracy.item(), 2))
        # backdoor accuracy
        if config['FL']['is_attack']:
            # and rd >= config['Attack']['attack_round'][0]
            if config['Attack']['name'] == 'SemanticAttack':
                test_attack_accuracy, _ = attacker.eval(dataset_train, global_model)
            else:
                # DBA
                test_attack_accuracy, _ = attacker.eval(dataset_test, global_model)
            logger.info("Attack accuracy: {:.2f}%".format(test_attack_accuracy))
            BA.append(round(test_attack_accuracy.item(), 2))

    # save the select info
    info_path = os.path.join("./save/historical_information", config['Dataset']['name'])
    save_select_info(info_path, select_info)
    # save the training loss info
    save_train_loss(info_path, round_losses)

    logger.debug(f'Main Accuracy:{MA}')
    logger.debug(f'Backdoor Accuracy:{BA}')

    # save the final global model
    if config['FL']['save_final_model']:
        save_final_model_path = os.path.join('./save/final_model/', config['Dataset']['name'])
        if not os.path.exists(save_final_model_path):
            os.makedirs(save_final_model_path)
        torch.save(global_model.state_dict(), save_final_model_path +
                   '/{}_{}.pth'.format(config['Model']['name'], args.config))
        logger.info('The final global model has been saved')

    # plot training loss curve
    if config['FL']['plot_loss_curve']:
        plt.figure()
        plt.plot(range(len(round_losses)), round_losses)
        plt.ylabel('train_loss')
        save_loss_curve_path = os.path.join('./save/loss_curve/', config['Dataset']['name'])
        if not os.path.exists(save_loss_curve_path):
            os.makedirs(save_loss_curve_path)
        plt.savefig(save_loss_curve_path + f'/{args.config}.png')
        logger.info('The training loss_curve curve has been saved')
