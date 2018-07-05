import json
import argparse

from ncg.predict import predict

from filepaths import fpaths_image_split
from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist

	
def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Predict image descriptions using the trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict_opts(parser)
    opt = parser.parse_args()
    return opt

def predict_opts(parser):
    parser.add_argument(
        '--config', 
        help = "Path to config file in JSON format",
        default = 'config_demo.json')
    # TODO: allow overwriting config with commandline arguments
    
def load_config(fpath_config):
    with open(fpath_config) as f:
        config = json.load(f)
    return config['predict']

def predict_image_descriptions(config):
    fpaths_images = fpaths_image_split(
        config['dir_images'], config['fpath_image_split'], is_encoded = False)
    fpath_decoder = config['fpath_decoder']
    fpath_vocab = config['fpath_vocab']
    fpath_save_predictions = config['fpath_save_predictions']
    encoder_model = config['encoder_model']
    encoder_layer = config['encoder_layer']
    max_length = config['max_length']
    dl_params = config['dl_params']

    check_files_not_exist([fpath_save_predictions]) 
    ensure_paths_exist([fpath_save_predictions])

    predict(fpaths_images, fpath_decoder, fpath_vocab,
            fpath_save_predictions, encoder_model, encoder_layer,
            max_length, dl_params)
        
def main():
    opt = parse_args()
    config = load_config(opt.config)
    predict_image_descriptions(config)


if __name__ == "__main__":
    main()
     
      

