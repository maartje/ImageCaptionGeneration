"""
Tests for training the neural models
"""

import unittest
import mock # mock file access
#import torch

from ncg.train import train

from test.test_helpers import MockEmbeddingDescriptionDataset


class TestTrain(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('torch.save')
    @mock.patch('ncg.train.EmbeddingDescriptionGroupedDataset')    
    def test_train(self, ds_class, torch_save, prnt = None):
        fpaths_images_train = ['im1.pt', 'im2.pt', 'im3.pt']
        fpaths_captions_train = ['c1.pt', 'c2.pt']
        ds_class.return_value = MockEmbeddingDescriptionDataset(fpaths_images_train, fpaths_captions_train)
        vocab_size = MockEmbeddingDescriptionDataset.vocab_size
        encoding_size = MockEmbeddingDescriptionDataset.encoding_size
        
        fpath_loss_data = "losses.pt"
        fpath_decoder = "decoder.pt"
        train(fpaths_images_train, fpaths_captions_train,
              [], [], # TODO: mock train and val using side effect, and related data
              vocab_size, encoding_size, 
              fpath_loss_data, fpath_decoder,
              learning_rate = 0.8, max_epochs = 15, dl_params = {}, store_loss_every = 1,
              print_loss_every = 3)

