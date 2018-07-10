import os
import glob
import torch

from parse_config import get_configuration

from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist

from ncg.train import train
from ncg.io.file_helpers import read_lines

def train_model(config):
    # file paths images 
    dir_image_encodings = config['dir_image_encodings']
    fpath_images_train = config['fpath_images_train']
    fnames_images_train = read_lines(fpath_images_train)
    fpaths_images_train = [
        os.path.join(dir_image_encodings, f'{fname}.pt') for fname in fnames_images_train]
    fpath_images_val = config['fpath_images_val']
    fnames_images_val = read_lines(fpath_images_val)
    fpaths_images_val = [
        os.path.join(dir_image_encodings, f'{fname}.pt') for fname in fnames_images_val]

    # file paths captions
    fpattern_captions_train = config['fpattern_captions_train']
    fpaths_captions_train = glob.glob(fpattern_captions_train)
    fpattern_captions_val = config['fpattern_captions_val']
    fpaths_captions_val = glob.glob(fpattern_captions_val)

    # vocab size, encoding size
    fpath_vocab = config['fpath_vocab']
    vocab = torch.load(fpath_vocab)
    vocab_size = vocab.vocab.n_words
    im1 = torch.load(fpaths_images_train[0])
    encoding_size = im1.size()[0]

    # fpaths to save model and loss info
    fpath_save_losses = config['fpath_save_losses']
    fpath_save_decoder = config['fpath_save_decoder']

    # training params
    learning_rate = config.get('learning_rate', 0.1)

    # others
    store_loss_every = config.get('store_loss_every', 1000)
    max_epochs = config.get('max_epochs', 30)
    max_hours = config.get('max_hours', 72)
    dl_params = {} 

#    print(fpaths_images_train[:2])
#    print(fpaths_captions_train[:2])
#    print(fpaths_images_val[:2])
#    print(fpaths_captions_val[:2])
#    print(vocab_size)
#    print(encoding_size)
#    print(fpath_save_losses)
#    print(fpath_save_decoder)
#    print(learning_rate)
#    print(max_epochs)
#    print(dl_params)
#    print(store_loss_every)

    check_files_exist(fpaths_images_train + fpaths_images_val)
    check_files_not_exist([fpath_save_losses, fpath_save_decoder])
    ensure_paths_exist([fpath_save_losses, fpath_save_decoder])
    
    train(fpaths_images_train, fpaths_captions_train,
          fpaths_images_val, fpaths_captions_val, 
          vocab_size, encoding_size,
          fpath_save_losses, fpath_save_decoder,
          max_train_instances = None, #TODO pass into dataset
          learning_rate = learning_rate, 
          max_epochs = max_epochs, max_hours = max_hours, 
          dl_params = dl_params, store_loss_every = store_loss_every)

     
def main():
    config = get_configuration('train', 
                               description = 'Train model for generating image descriptions.')
    train_model(config)


if __name__ == "__main__":
    main()
     
      
