import argparse
import configparser
import torch

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/STID/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--device', type=str, default=config['general']['device'])

    # Data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_len', type=int, default=config['data']['input_len'])
    parser.add_argument('--output_len', type=int, default=config['data']['output_len'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])

    # Model
    parser.add_argument('--input_dim', type=int, default=config['model']['input_dim'])
    parser.add_argument('--node_dim', type=int, default=config['model']['node_dim'])
    parser.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'])
    parser.add_argument('--num_layer', type=int, default=config['model']['num_layer'])
    parser.add_argument('--temp_dim_tid', type=int, default=config['model']['temp_dim_tid'])
    parser.add_argument('--temp_dim_diw', type=int, default=config['model']['temp_dim_diw'])
    parser.add_argument('--time_of_day_size', type=int, default=config['model']['time_of_day_size'])
    parser.add_argument('--day_of_week_size', type=int, default=config['model']['day_of_week_size'])

    # Features Flags
    parser.add_argument('--if_time_in_day', type=eval, default=config['model']['if_time_in_day'])
    parser.add_argument('--if_day_in_week', type=eval, default=config['model']['if_day_in_week'])
    parser.add_argument('--if_spatial', type=eval, default=config['model']['if_spatial'])

    args, _ = parser.parse_known_args()
    return args