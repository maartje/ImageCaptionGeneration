import json
import argparse
from filepaths import get_file_paths

def parse_args(section, description):
    parser = argparse.ArgumentParser(
        description = description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options(parser, section)
    opt = parser.parse_args()
    return opt

def options(parser, section):
    parser.add_argument(
        '--config', 
        help = "Path to config file in JSON format",
        default = 'config_demo.json')
    parser.add_argument(
        '--main_dir', 
        help = "Overwrites the config setting for the main directory containing input and output data")
    parser.add_argument(
        '--model_dir', 
        help = "Overwrites the config setting for the model and train regime specific output")
    if section == 'train':
        parser.add_argument(
            '--learning_rate', 
            type = float,
            help = "Overwrites the config setting for the learning rate")
    
def load_config(fpath_config):
    with open(fpath_config) as f:
        config = json.load(f)
    return config

def get_configuration(section, description = ''):
    opt = parse_args(section, description)
    config = load_config(opt.config)
    # TODO: allow overwriting config settings with commandline arguments
    
    opt_dict = vars(opt)
    opt_dict = {k : v for k, v in opt_dict.items() if not (v is None)}
    config['general'].update({ k:v for k, v in opt_dict.items() if k in config['general']})
    config[section].update({ k:v for k, v in opt_dict.items() if k in config[section]})
    
    filepaths = get_file_paths(config['general'])
    return config[section], filepaths

