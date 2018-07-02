import json
import argparse

from helpers import check_infiles_exists, check_outfiles_exists, ensure_fpath_exists

from ncg.report import plot_losses

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Generate plots and scores for evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    report_opts(parser)
    opt = parser.parse_args()
    return opt

def report_opts(parser):
    parser.add_argument(
        '--config', 
        help = "Path to config file in JSON format",
        default = 'config_demo.json')
    # TODO: allow overwriting config with commandline arguments
    
def load_config(fpath_config):
    with open(fpath_config) as f:
        config = json.load(f)
    return config['report']

def report(config):
    fpath_epoch_loss = config['fpath_plot_epoch_loss']
    fpath_batch_loss = config['fpath_plot_batch_loss']
    if check_outfiles_exists([fpath_epoch_loss, fpath_batch_loss]):
        return
    ensure_fpath_exists(fpath_epoch_loss)
    ensure_fpath_exists(fpath_batch_loss)

    if config['fpath_losses']:
        plot_losses(
            config['fpath_losses'], 
            fpath_epoch_loss, 
            fpath_batch_loss)
   
def main():
    opt = parse_args()
    config = load_config(opt.config)
    report(config)


if __name__ == "__main__":
    main()
     
      

