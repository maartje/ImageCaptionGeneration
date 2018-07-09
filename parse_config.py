import json
import argparse

def parse_args(description):
    parser = argparse.ArgumentParser(
        description = description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options(parser)
    opt = parser.parse_args()
    return opt

def options(parser):
    parser.add_argument(
        '--config', 
        help = "Path to config file in JSON format",
        default = 'config_demo.json')
    
def load_config(fpath_config):
    with open(fpath_config) as f:
        config = json.load(f)
    return config

def get_configuration(section, description = ''):
    opt = parse_args(description)
    config = load_config(opt.config)
    # TODO: allow overwriting config settings with commandline arguments
    return config[section]

