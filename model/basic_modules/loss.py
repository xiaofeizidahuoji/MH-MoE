import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lib.metrics import MAE_torch_all

def mem_class_loss(pred, all_x_predict_ori, gate, label, scaler=None, topk=3):
    n_model = all_x_predict_ori.shape[0]
    if scaler:
        pred = scaler.inverse_transform(pred)
        label = scaler.inverse_transform(label)
        all_n_pred = scaler.inverse_transform(all_x_predict_ori) # [n_pred,B,T,N,1]
    _, _, all_mae = MAE_torch_all(all_n_pred, label[None,...].expand(n_model,-1,-1,-1,-1), 0.001) # [n,B,T,N,1]

    mae_mean = all_mae.mean(dim=2)  # [n_model, B, N, 1]
    hard_labels = torch.argmin(mae_mean,dim=0).view(-1) # [B,N,1]
    loss_hard = F.nll_loss(torch.log(gate.view(-1,n_model)), hard_labels)
    return 5*loss_hard