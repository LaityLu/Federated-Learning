import argparse
import copy

import torch
import yaml
import os

import attack
import dataloader
import model
import recover
import trainer
from utils import check, load_global_model, load_client_model, load_clients_data, \
    load_select_info, load_train_loss, setup_logger, load_aggr_clients

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
    logger = setup_logger(log_file_path + '/{}_{}.log'.format(args.config, config['Recover']['name']))

    # get dataset
    dataset_train, dataset_test = getattr(dataloader, config['Dataset']['name'])()
    clients_data_dir_path = os.path.join('./save/data_sampled', config['Dataset']['name'])
    # check whether that data has been sampled
    assert (check(clients_data_dir_path + '/data_config.json', config)), "No correct sampled client data"
    # load client data
    dict_clients, list_num_dps = load_clients_data(clients_data_dir_path)

    # get the model
    original_model = getattr(model, config['Model']['name'])(**config['Model']['args'])
    original_model.to(device)

    # get the trainer
    # normal trainer
    local_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'],
                                                                device=device)
    # it's for testing backdoor accuracy after recover
    if config['FL']['is_attack']:
        # malicious trainer
        attacker = getattr(attack, config['Attack']['name'])(**config['Attack']['args'],
                                                             adversary_list=config['Attack']['adversary_list'],
                                                             device=device)

    # load the historical information
    # load the info of train process
    select_info_path = os.path.join("./save/historical_information", config['Dataset']['name'])
    select_info = load_select_info(select_info_path)
    train_loss = load_train_loss(select_info_path)
    # load the benign clients that defense algorithm chose
    aggr_clients = load_aggr_clients(select_info_path)
    # load the malicious client id / unlearned client id
    malicious_clients = config['Recover']['malicious_clients']
    # load the models
    old_global_models = []
    old_client_models = []
    for rd in range(config['FL']['round']):
        model_state_dict = torch.load(
            load_global_model(config['Dataset']['name'], rd))
        original_model.load_state_dict(model_state_dict)
        old_global_models.append(copy.deepcopy(original_model))
        model_state_dict_ = torch.load(
            load_client_model(config['Dataset']['name'], rd))
        temp_client_models = []
        for i in range(int(config['FL']['num_clients'] * config['FL']['frac'])):
            original_model.load_state_dict(model_state_dict_[i])
            temp_client_models.append(copy.deepcopy(original_model))
        old_client_models.append(temp_client_models)
    model_state_dict = torch.load(
        load_global_model(config['Dataset']['name'], config['FL']['round']))
    original_model.load_state_dict(model_state_dict)
    old_global_models.append(copy.deepcopy(original_model))

    # Recover process
    recover_maker = getattr(recover, config['Recover']['name'])(dataset_train, dataset_test, dict_clients,
                                                                list_num_dps, select_info, malicious_clients,
                                                                train_loss=train_loss,
                                                                aggr_clients=aggr_clients,
                                                                **config['Recover']['args'])
    # get the recovered global model
    recovered_global_model_state = recover_maker.recover(local_trainer, old_global_models, old_client_models)
    original_model.load_state_dict(recovered_global_model_state)

    # test
    # backdoor accuracy
    if config['FL']['is_attack']:
        if config['Attack']['name'] == 'SemanticAttack':
            test_attack_accuracy, _ = attacker.eval(dataset_train, original_model)
        else:
            # DBA
            test_attack_accuracy, _ = attacker.eval(dataset_test, original_model)
        logger.info("Backdoor accuracy: {:.2f}%".format(test_attack_accuracy))
