"""
Tests for training the neural models
"""

import unittest
import mock # mock file access

from ncg.train import train

from test.test_helpers import MockImageCaptionDataset


class TestTrain(unittest.TestCase):

    @mock.patch('ncg.train.ImageCaptionDataset')    
    def test_train(self, ds_class):
        fpaths_image_encodings = ['im1.pt', 'im2.pt', 'im3.pt']
        fpaths_captions = ['c1.pt', 'c2.pt']
        ds_class.return_value = MockImageCaptionDataset(fpaths_image_encodings, fpaths_captions)
        vocab_size = MockImageCaptionDataset.vocab_size
        encoding_size = MockImageCaptionDataset.encoding_size
        
        train(fpaths_image_encodings, fpaths_captions, vocab_size, encoding_size, 
              learning_rate = 0.8)

