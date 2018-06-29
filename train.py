import os
import glob
import torch

from ncg.train import train
from ncg.io.file_helpers import read_lines

dir_images = os.path.join('output', 'demo', 'resnet18_avgpool')
fpath_image_splits_train = os.path.join('data', 'demo', 'image_splits', 'train_images.txt')
fnames_images_train = read_lines(fpath_image_splits_train)
fpaths_images_train = [os.path.join(dir_images, f'{fname}.pt') for fname in fnames_images_train]
fpath_image_splits_val = os.path.join('data', 'demo', 'image_splits', 'val_images.txt')
fnames_images_val = read_lines(fpath_image_splits_val)
fpaths_images_val = [os.path.join(dir_images, f'{fname}.pt') for fname in fnames_images_val]

fpattern_captions_train = os.path.join('output', 'demo', "train.[0-9].en.pt")
fpaths_captions_train = glob.glob(fpattern_captions_train)

fpattern_captions_val = os.path.join('output', 'demo', "val.[0-9].en.pt")
fpaths_captions_val = glob.glob(fpattern_captions_val)

vocab = torch.load(os.path.join('output', 'demo', 'vocab.pt'))
vocab_size = vocab.vocab.n_words

im1 = torch.load(fpaths_images_train[0])
encoding_size = im1.size()[0]

fpath_loss_data_out = os.path.join('output', 'demo', "loss.pt") 
fpath_decoder_out = os.path.join('output', 'demo', "show_tell.pt")

train(fpaths_images_train, fpaths_captions_train,
      fpaths_images_val, fpaths_captions_val, 
      vocab_size, encoding_size,
      fpath_loss_data_out, fpath_decoder_out,
      max_train_instances = None, #TODO pass into dataset
      learning_rate = 0.1, max_epochs = 10, dl_params = {}, store_loss_every = 100)
