import torch


def process_train_intermoe_keys(state_dict, model_name, args):
    # moe
    if model_name == 'GWN':
        state_dict = {k: v for k, v in state_dict.items() if 'start_conv' not in k}
        state_dict = {k.replace('.0', '') if '_pretrain.0' in k else k: v for k, v in state_dict.items()}
    elif model_name == 'MTGNN':
        state_dict = {k: v for k, v in state_dict.items() if 'start_conv' not in k and 'skip0' not in k}
        state_dict = {k.replace('.0', '') if '_pretrain.0' in k and 'norm' not in k else k: v for k, v in state_dict.items()}
    elif model_name == 'PDFormer':
        state_dict = {k: v for k, v in state_dict.items() if 'enc_embed_layer.value_embedding.token_embed' not in k}
    elif model_name == 'STID':
        state_dict = {k: v for k, v in state_dict.items() if 'time_series_emb_layer' not in k}
    elif model_name == 'STAEFormer':
        state_dict = {k: v for k, v in state_dict.items() if 'input_proj' not in k}
    else:
        return state_dict
    return state_dict


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'true' or 'false'")
    
    