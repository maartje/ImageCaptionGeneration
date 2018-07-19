"""
Tests for training the neural models
"""

import unittest
import mock # mock file access
#import torch

from ncg.train import train

from test.test_helpers import MockEmbeddingDescriptionDataset

def mock_torch_load(fpath):
    if fpath == fpath_vocab:
        return 

class MockTextMapper:
    def __init__(self):

fpath_vocab = 'vocab.pt'
fpaths_captions_train = ['c1.pt', 'c2.pt']

class TestTrain(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('torch.load', side_effect = mock_torch_load)
    @mock.patch('torch.save')
    @mock.patch('ncg.train.ImageFeaturesDescriptionsDataset')    
    @mock.patch('ncg.train.ImageFeaturesDataset')    
    def test_train(self, ds_imfeats_class, ds_class, torch_save, torch_load, prnt = None):
        fpaths_images_train = ['im1.pt', 'im2.pt', 'im3.pt']
        ds_class.return_value = MockEmbeddingDescriptionDataset(
            fpaths_images_train, fpaths_captions_train)
        ds_imfeats_class = [ed[0] for ed in ds_class.return_value]
        vocab_size = MockEmbeddingDescriptionDataset.vocab_size
        encoding_size = MockEmbeddingDescriptionDataset.encoding_size
        
        fpath_loss_data = "losses.pt"
        fpath_decoder = "decoder.pt"
        train(fpaths_images_train, fpaths_captions_train,
              fpaths_images_train, fpaths_captions_train, 
              512, fpath_vocab, 20, 
              fpath_loss_data, fpath_decoder,
              learning_rate = 0.8, max_epochs = 15, dl_params_train = {'batch_size' : 2}, 
              print_loss_every = 3)

