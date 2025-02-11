import argparse
import copy
from datetime import datetime

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
from utils.logger_config import logger

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./params.yaml', help='the path of config file')
    args = parser.parse_args()
    # load the config file
    try:
        with open(f'./{args.config}', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"can't find {args.config}")
    device = torch.device('cuda:{}'.format(config['gpu'])
                          if torch.cuda.is_available() and config['gpu'] != -1 else 'cpu')
    # save time
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H-%M")
    logger.info(f'time:{formatted_time}')

    # get the dataset
    dataset_train, dataset_test = getattr(dataloader, config['Dataset']['name'])()

    # get the model
    global_model = getattr(model, config['Model']['name'])(**config['Model']['args'])
    global_model.to(device)

    # sampling data
    sampler = getattr(sampler, config['Sampler']['name'])(dataset_train,
                                                          **config['Sampler']['args'],
                                                          num_clients=config['FL']['num_clients'],
                                                          is_attack=config['FL']['is_attack'],
                                                          poison_images=config['Attack']['poison_images'])
    dict_clients, list_num_dps = sampler.sample()

    # fed training
    round_losses = []
    for rd in range(config['FL']['round']):
        # select clients
        num_select_clients = max(int(config['FL']['frac'] * config['FL']['num_clients']), 1)
        select_clients = np.random.choice(range(config['FL']['num_clients']), num_select_clients, replace=False)
        num_dps = [list_num_dps[i] for i in select_clients]
        # begin
        print("-----  Round {:3d}  -----".format(rd))
        logger.info("Round {:3d}:".format(rd))
        # store the local loss and local model for each client
        locals_losses = []
        local_models = []
        have_attack = False
        for idxes in select_clients:
            # confrontation model training
            if config['FL']['is_attack'] and rd in config['Attack']['attack_round'] and not have_attack:
                logger.info('malicious client:{}'.format(idxes))
                attacker = getattr(attack, config['Attack']['name'])(**config['Attack']['args'],
                                                                     poison_images=config['Attack']['poison_images'],
                                                                     device=device)
                local_model, local_loss = attacker.exec(dataset_train, dict_clients[idxes],
                                                        copy.deepcopy(global_model))
                have_attack = True
            # normal model training
            else:
                local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                            device=device)
                local_model, local_loss = local_trainer.update(dataset_train, dict_clients[idxes],
                                                               copy.deepcopy(global_model))
            local_models.append(local_model)
            locals_losses.append(local_loss)
        # defense
        if config['FL']['is_defense']:
            defender = getattr(defense, config['Defense']['name'])(**config['Defense']['args'])
            benign_clients, malicious_clients = defender.exec(global_model.state_dict(), local_models,
                                                              select_clients)
            bn_local_models = [local_models[i] for i in [select_clients.tolist().index(v) for v in benign_clients]]
            bn_num_dps = [list_num_dps[i] for i in benign_clients]
            global_model_state_dict = getattr(aggregator, config['FL']['aggregator'])(bn_local_models, bn_num_dps)
        # normal aggregation
        else:
            global_model_state_dict = getattr(aggregator, config['FL']['aggregator'])(local_models, num_dps)

        # update global model
        global_model.load_state_dict(global_model_state_dict)

        # compute the average loss in a round
        round_loss = sum(locals_losses) / len(locals_losses)
        print('Training average loss {:.3f}'.format(round_loss))
        logger.info('Training average loss {:.3f}'.format(round_loss))
        round_losses.append(round_loss)

        # testing
        local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                    device=device)
        # train_accuracy, train_loss = local_trainer.eval(dataset_train, global_model)
        # print("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))
        # logger.info("Training accuracy: {:.2f}%, loss: {:.3f}".format(train_accuracy, train_loss))

        test_accuracy, test_loss = local_trainer.eval(dataset_test, global_model)
        print("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
        logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))

        if config['FL']['is_attack']:
            attacker = getattr(attack, config['Attack']['name'])(**config['Attack']['args'],
                                                                 poison_images=config['Attack']['poison_images'],
                                                                 device=device)
            test_attack_accuracy, _ = attacker.eval(dataset_train, global_model)
            print("Attack accuracy: {:.2f}%".format(test_attack_accuracy))
            logger.info("Attack accuracy: {:.2f}%".format(test_attack_accuracy))

    # save the final global model
    torch.save(global_model.state_dict(), './save/save_model/{}_{}.pth'.format(config['Model']['name'], formatted_time))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(round_losses)), round_losses)
    plt.ylabel('train_loss')
    plt.savefig('./save/loss/{}.png'.format(formatted_time))
