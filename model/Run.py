import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import random
from model.Trainer_moe import Trainer_Moe
from MHMoE import MHMoE as Network_train
from MHMoE import MHMoE as Network_Predict
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters, print_require_grad_parameters
from lib.metrics import MAE_torch
from lib.predifineGraph import *
from lib.utils import *
from lib.data_process import define_dataloder, get_val_tst_dataloader, data_type_init
from conf.MHMoE.Params_train import parse_args_moe
import torch.nn.functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args_moe(device)
# preprocess the args
args.model_list = list(args.model_list.split(','))
args.load_train_paths = list(args.load_train_paths.split(','))
args.sparse_qk = str_to_bool(args.sparse_qk)

print('Mode: ', args.mode, '  model_list: ', args.model_list, '  DATASET: ', args.dataset_use,
      '  load_train_paths: ', args.load_train_paths)

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def scaler_mae_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        # print(mae.shape, mae_loss.shape)
        return mae, mae_loss
    return loss


if 'GWN' in args.model_list or 'MTGNN' in args.model_list:
    seed_mode = False   # for quick running
else:
    seed_mode = True
init_seed(args.seed, seed_mode)

#config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
if args.mode == 'test':  # use id in training
    log_dir = os.path.join(current_dir, '../SAVE', args.mode+'_moe', '_'.join(args.model_list), str(args.exp_id))
else:
    rand_id = int(random.SystemRandom().random() * 100000) 
    log_dir = os.path.join(current_dir, '../SAVE', args.mode+'_moe', '_'.join(args.model_list), str(rand_id))
Mkdir(log_dir)
args.log_dir = log_dir

#predefine Graph
args.dataset_graph = [args.dataset_use]
pre_graph_dict(args)
data_type_init(args.dataset_use, args)

args.xavier = False

#load dataset
if args.mode == 'train':
    x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, _, _, scaler_dict = define_dataloder(stage='Train', args=args)
    eval_train_loader, eval_val_loader, eval_test_loader, eval_scaler_dict = None, None, None, None
    _,_, eval_x_val_dict, eval_y_val_dict, eval_x_tst_dict, eval_y_tst_dict, eval_scaler_dict = define_dataloder(stage='eval', args=args)
    eval_val_loader = get_val_tst_dataloader(eval_x_val_dict, eval_y_val_dict, args, shuffle=False)
    eval_test_loader = get_val_tst_dataloader(eval_x_tst_dict, eval_y_tst_dict, args, shuffle=False)
else:
    x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, scaler_dict = None, None, None, None, None
    eval_x_trn_dict, eval_y_trn_dict, eval_x_val_dict, eval_y_val_dict, eval_x_tst_dict, eval_y_tst_dict, eval_scaler_dict = define_dataloder(stage='eval', args=args)
    eval_train_loader = get_val_tst_dataloader(eval_x_trn_dict, eval_y_trn_dict, args, shuffle=True)
    eval_val_loader = get_val_tst_dataloader(eval_x_val_dict, eval_y_val_dict, args, shuffle=False)
    eval_test_loader = get_val_tst_dataloader(eval_x_tst_dict, eval_y_tst_dict, args, shuffle=False)


#init model
if args.mode == 'train':
    model = Network_train(args)
    for cnt, model_name in enumerate(args.model_list):
        if 'STID' in model_name:
            model_name1 = 'STID'
        elif 'GWN' in model_name:
            model_name1 = 'GWN'
        elif 'STWave' in model_name:
            model_name1 = 'STWave'
        else:
            model_name1 = model_name
        load_dir = os.path.join(current_dir, '../experts', model_name1)
        state_dict = torch.load(load_dir + '/' + args.load_train_paths[cnt])
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(f'predictors.{model_name}.', '')
            new_state_dict[new_key] = value
        train_state_dict = process_train_intermoe_keys(new_state_dict, model_name, args)
        model.predictors[model_name].load_state_dict(train_state_dict, strict=False)
        print(load_dir + '/' + args.load_train_paths[cnt])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(args.device)
    print('load train model for train_moe!!!')
else:
    model = Network_Predict(args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(args.device)
    if args.mode == 'eval' or args.mode == 'test':
        load_dir = os.path.join(current_dir, '../SAVE', 'train_moe', '_'.join(args.model_list))
        model_weights = torch.load(load_dir + '/'+ str(args.exp_id) + '/' + args.load_train_path)
        model_weights = {k.replace('.0', '') if '_train.0' in k and 'norm' not in k else k: v for k, v in model_weights.items()}
        model.load_state_dict(model_weights, strict=True)
        print(load_dir + '/' + args.load_train_path)
        print('load train model!!!')

print_model_parameters(model, only_num=False)
print_require_grad_parameters(model)

#init loss function, optimizer
loss = scaler_mae_loss(mask_value=args.mape_thresh)
print('============================scaler_mae_loss')
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)


#start training
trainer = Trainer_Moe(model, loss, optimizer, x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, args.A_dict, args.lpls_dict, eval_train_loader,
                       eval_val_loader, eval_test_loader, scaler_dict, eval_scaler_dict, args,
                       )
print(args)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    trainer.test_moe(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader, eval_scaler_dict[args.dataset_use], 
                               trainer.logger)
else:
    raise ValueError
