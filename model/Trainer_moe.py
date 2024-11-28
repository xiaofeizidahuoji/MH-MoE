import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
from lib.data_process import get_train_task_batch
from lib.TrainInits import print_model_parameters, print_require_grad_parameters

from model.basic_modules import mem_class_loss

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

class Trainer_Moe(object):
    def __init__(self, model, loss, optimizer, x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, A_dict, lpls_dict, eval_train_dataloader,
                       eval_val_dataloader, eval_test_dataloader, scaler_dict, eval_scaler_dict,
                  args, lr_scheduler=None):
        super(Trainer_Moe, self).__init__()
        self.model = model
        self.model_name = args.model_list[0]
        self.args = args
        self.loss = loss
        self.loss_class = mem_class_loss
        self.optimizer = optimizer
        self.x_trn_dict, self.y_trn_dict = x_trn_dict, y_trn_dict
        self.A_dict, self.lpls_dict = A_dict, lpls_dict
        self.eval_train_loader = eval_train_dataloader
        self.eval_val_loader = eval_val_dataloader
        self.eval_test_loader = eval_test_dataloader
        self.scaler_dict = scaler_dict
        self.eval_scaler_dict = eval_scaler_dict
        self.lr_scheduler = lr_scheduler
        # mem
        self.warmup_steps=5
        self.plateau_steps = 30 # 30
        self.lr = args.lr_init
        self.min_lr = 3e-5 # 3e-5
        self.lr_anneal_steps = self.args.train_epochs 
        
        if eval_train_dataloader is not None:
            self.eval_train_per_epoch = len(eval_train_dataloader)
            self.eval_val_per_epoch = len(eval_val_dataloader)
        self.batch_seen = 0
        self.best_path = os.path.join(self.args.log_dir)
        Mkdir(self.best_path)
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name='', debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    # scheduler
    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.warmup_steps + self.plateau_steps:
            return self.lr
        elif self.step < self.lr_anneal_steps + self.plateau_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps - self.plateau_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


    # moe+mem train
    def train(self, ):
        """
        training stage
        """
        best_loss = float('inf')
        not_improved_count = 0
        val_losses = []  # 保存每个 epoch 的 val_loss
        for epoch in tqdm(range(self.args.train_epochs)):
            if epoch==self.args.explore_stage:
                for train_model in self.model.model_list:
                    for param in self.model.predictors[train_model].parameters():
                        param.requires_grad = False
                self.optimizer = torch.optim.Adam(params=self.model.parameters(), eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
                print_model_parameters(self.model, only_num=False)
                print_require_grad_parameters(self.model)
            self.step = epoch
            start_time = time.time()
            spt_task_x, spt_task_y, select_dataset, train_len = get_train_task_batch(self.args, self.x_trn_dict, self.y_trn_dict)
            print(select_dataset)
            loss, loss_pred = self.train_eps_mem(spt_task_x, spt_task_y, select_dataset, train_len, epoch)
            val_loss = self.eval_eps_mem()
            val_losses.append(val_loss)
            end_time = time.time()
            if epoch % 1 == 0:
                print(
                    "[train] epoch #{}/{}: loss is {} pred loss {}, val_loss is {}, lr {}, training time is {}".format(
                    epoch + 1, self.args.train_epochs, round(loss, 2), round(loss_pred, 2), round(val_loss, 2),
                    self.optimizer.param_groups[0]['lr'], round(end_time - start_time, 2)))
                if val_loss < best_loss:
                    best_loss = val_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save best state
            if best_state == True:
                best_premodel = copy.deepcopy(self.model.state_dict())
                save_path = self.best_path+'/'+select_dataset+'_ep'+str(epoch)+'_EL'+str(round(loss, 2))+'_PL'+str(round(loss_pred, 2))+'.pth'
                torch.save(best_premodel, save_path)

            self.logger.info("Saving current best model to " + self.best_path)
        print("Pre-train finish.")
        # test
        self.model.load_state_dict(best_premodel, strict=True)
        self.test_moe(self.model, self.args, self.args.A_dict, self.args.lpls_dict, self.eval_test_loader, self.eval_scaler_dict[self.args.dataset_use], 
                               self.logger)
        # save val_losses
        with open(os.path.join(self.best_path, "val_losses.txt"), "w") as f:
            for epoch, loss in enumerate(val_losses):
                f.write(f"Epoch {epoch + 1}: Val Loss = {loss}\n")
        print("Pre-train finish.")
    
    def train_eps_mem(self, spt_task_x, spt_task_y, select_dataset, train_len, epoch):
        self.model.train()
        total_loss = 0
        pred_loss = 0
        
        with torch.autograd.set_detect_anomaly(True):
            nadj = self.A_dict[select_dataset]
            lpls = self.lpls_dict[select_dataset]
            if torch.cuda.device_count() > 1:
                lpls = lpls.unsqueeze(0).expand(torch.cuda.device_count(),-1,-1).to(self.args.device)
                nadj = nadj.unsqueeze(0).expand(torch.cuda.device_count(),-1,-1).to(self.args.device)
            pbar = tqdm(range(train_len))
            for i in pbar:
                x_in, y_in, y_lbl = spt_task_x[i], spt_task_y[i], spt_task_y[i][..., 0:1]
                x_in, y_in, y_lbl = x_in.to(self.args.device), y_in.to(self.args.device), y_lbl.to(self.args.device)
                out, all_x_predict_ori, label_retrived = self.model(x_in, y_in, select_dataset, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=True, epoch=epoch)
                ############## loss ##############
                loss_pred, _ = self.loss(out[...,:1], y_lbl, self.scaler_dict[select_dataset])
                
                loss_class = self.loss_class(out, all_x_predict_ori, label_retrived, y_lbl, self.scaler_dict[select_dataset], self.args.topk)
                
                loss = loss_pred + loss_class
                pbar.set_description(f'loss {loss}, pred loss {loss_pred}, class loss {loss_class}')

                self.optimizer.zero_grad()
                loss.backward()
                
                ############ lr_sceduler & grad_norm ####################
                self._anneal_lr()
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                total_loss += loss.item()
                pred_loss += loss_pred.item()

        train_epoch_loss = total_loss / train_len
        train_pred_loss = pred_loss / train_len

        return train_epoch_loss, train_pred_loss
    
    def eval_eps_mem(self, ):
        self.model.eval()
        total_val_loss = 0
        nadj = self.A_dict[self.args.dataset_use]
        lpls = self.lpls_dict[self.args.dataset_use]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.eval_val_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                out, all_x_predict_ori, label_retrived = self.model(data, data, self.args.dataset_use, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=False)
                loss_pred, _ = self.loss(out[...,:1], label[..., :1], self.eval_scaler_dict[self.args.dataset_use])
                loss = loss_pred
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()  
        val_epoch_loss = total_val_loss / len(self.eval_val_loader)
        
        return val_epoch_loss
    
    ######### ori ##############
    def train_eval(self):
        """
        prompt-tunning stage
        """
        val_losses = []
        
        self.lr_anneal_steps = self.args.ori_epochs
        self.plateau_steps = 60
        
        best_loss = float('inf')
        not_improved_count = 0
        if self.args.mode == 'eval':
            eps = self.args.eval_epochs
        else:
            eps = self.args.ori_epochs
        for epoch in tqdm(range(eps)):
            self.step = epoch
            start_time = time.time()
            train_epoch_loss, loss_pre = self.eval_trn_eps()
            end_time = time.time()
            print('time cost: ', round(end_time-start_time, 2))
            val_epoch_loss = self.eval_val_eps()
            print("[Target Fine-tune] epoch #{}/{}: loss is {}, val_loss is {}, lr:{}".format(
                epoch+1, eps, round(train_epoch_loss, 2), round(val_epoch_loss, 2), self.optimizer.param_groups[0]['lr'])
                )
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            val_losses.append(val_epoch_loss)

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

            # save the best state
            if best_state == True:
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, os.path.join(self.args.log_dir, self.args.model_list[0] + '_' + self.args.dataset_use + f'_epoch{epoch+1}'+ '.pth'))
                self.logger.info('*********************************Current best model saved!')

        with open(os.path.join(self.args.log_dir, "val_losses.txt"), "w") as f:
            for epoch, loss in enumerate(val_losses):
                f.write(f"Epoch {epoch + 1}: Val Loss = {loss}\n")
        self.eval_test(self.model, self.args, self.A_dict, self.lpls_dict, self.eval_test_loader, self.eval_scaler_dict[self.args.dataset_use], self.logger, best_path=self.best_path)

    def eval_trn_eps(self):
        self.model.train()
        total_loss = 0
        nadj = self.A_dict[self.args.dataset_use]
        lpls = self.lpls_dict[self.args.dataset_use]
        for batch_idx, (data, target) in enumerate(self.eval_train_loader):
            self.batch_seen += 1
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
            out, q = self.model(data, label, self.args.dataset_use, self.batch_seen, nadj=nadj, lpls=lpls, useGNN=True, DSU=True)
            loss_pred, _ = self.loss(out[..., :1], label[..., :1], self.eval_scaler_dict[self.args.dataset_use])
            
            if 'STWave' in self.args.model_list:
                loss_l, _ = self.loss(out[..., 1:2], out[..., 2:], None)
                loss = loss_l + loss_pred
            else:
                loss = loss_pred
            self.optimizer.zero_grad()
            loss.backward()
            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        train_epoch_loss = total_loss / self.eval_train_per_epoch

        if self.args.lr_decay:
            self.lr_scheduler.step(train_epoch_loss)
        return train_epoch_loss, loss_pred

    def eval_val_eps(self):
        self.model.eval()
        total_val_loss = 0
        nadj = self.A_dict[self.args.dataset_use]
        lpls = self.lpls_dict[self.args.dataset_use]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.eval_val_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                out, _ = self.model(data, data, self.args.dataset_use, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=False)
                loss, _ = self.loss(out[...,:1], label[...,:1], self.eval_scaler_dict[self.args.dataset_use])
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_epoch_loss = total_val_loss / self.eval_val_per_epoch
        return val_epoch_loss

    @staticmethod
    def eval_test(model, args, A_dict, lpls_dict, data_loader, scaler, logger, path=None, best_path=None): # 纯test
        nadj = A_dict[args.dataset_use]
        lpls = lpls_dict[args.dataset_use]
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(args.device)
                target = target.to(args.device)
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                output, _ = model(data, data, args.dataset_use, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=False)
                output = output[..., :args.output_dim]
                label = target[..., :args.output_dim]
                y_true.append(label)
                y_pred.append(output)
        if not args.real_value:
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        return mae



    ############ test ##############
    def test_moe(self, model, args, A_dict, lpls_dict, data_loader, scaler, logger): 
        nadj = A_dict[args.dataset_use]
        lpls = lpls_dict[args.dataset_use]
        model.eval()
        data_all = []
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Processing batches", leave=False):
                data = data.to(args.device)
                target = target.to(args.device)
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                output, all_x_predict_ori, label_retrived = model(data, data, args.dataset_use, batch_seen=None, nadj=nadj, 
                                      lpls=lpls, useGNN=True, DSU=False)
                
                label = target[..., :1]
                output = output[..., :1]
                y_true.append(label)
                y_pred.append(output)
            
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        if not args.real_value:
            data_all = scaler.inverse_transform(data_all)
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)
        
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        return mae
    