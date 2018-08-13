import os
import glob
import torch

from parse_config import get_configuration

from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist

from ncg.train import train
from ncg.io.file_helpers import read_lines

def train_model(config, filepaths):

#    print(filepaths['image_features_train'])
#    print(filepaths['caption_vectors_train'][:3])
#    print(filepaths['image_features_val'])
#    print(filepaths['caption_vectors_val'][:2])
#    print(filepaths['losses'])
#    print(filepaths['model'])
#    print(config['learning_rate'])
#    print(config['max_epochs'])
#    print(dl_params)
#    print(config['store_loss_every'])

    
    fpath_model_resume = None
    if config.get('fname_resume'):
        dir_model_resume = os.path.dirname(filepaths['model'])
        fpath_model_resume = os.path.join(dir_model_resume, config.get('fname_resume'))

    check_files_exist([filepaths['image_features_train'], 
                       filepaths['image_features_val'],
                       fpath_model_resume])
    #check_files_not_exist([filepaths['losses'], filepaths['model']])
    ensure_paths_exist([filepaths['losses'], filepaths['model']])
    
    train(filepaths['image_features_train'], filepaths['caption_vectors_train'],
          filepaths['image_features_val'], filepaths['caption_vectors_val'], 
          config['hidden_size'], filepaths['vocab'], config['max_length'],
          filepaths['losses'], filepaths['bleu_scores'], filepaths['model'], filepaths['best_model'],
          optimizer = config['optimizer'], learning_rate = config['learning_rate'], 
          lr_decay_start = config['lr_decay']['start'], lr_decay_end = config['lr_decay']['end'],
          max_epochs = config['max_epochs'], max_hours = config['max_hours'], 
          dl_params_train = config['dl_params_train'], dl_params_val = config['dl_params_val'],
          clip = config['clip'], fpath_out = filepaths.get('train_out'), 
          fpath_model_resume = fpath_model_resume)

     
def main():
    config, filepaths = get_configuration('train', 
                               description = 'Train model for generating image descriptions.')
    train_model(config, filepaths)


if __name__ == "__main__":
    main()
     
      
