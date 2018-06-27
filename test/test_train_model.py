"""
Tests the training of the models
"""

import unittest
import torch
import torch.nn as nn
from torch import optim

from ncg.nn.train_model import train_iter
from ncg.nn.show_and_tell import ShowAndTell


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.encoding_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_criterion = nn.NLLLoss()
       
    def test_train_iter_show_and_tell(self):
        decoder = ShowAndTell(self.encoding_size, self.vocab_size, self.device)
        optimizer = optim.SGD(decoder.decoder.parameters(), lr = 0.2)


        train_data = [self.generate_random_training_pair() for i in range(3)]
        losses = train_iter(decoder, train_data, self.loss_criterion, 
                                  optimizer, self.device, max_epochs = 10)        
        losses_1 = losses[::3] # losses for pair 1
        losses_2 = losses[1::3]
        losses_3 = losses[2::3]
        
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(is_decreasing(losses_1))
        self.assertTrue(is_decreasing(losses_2))
        self.assertTrue(is_decreasing(losses_3))

    def generate_random_training_pair(self):
        caption = torch.LongTensor(1, 10, device=self.device).random_(0, self.vocab_size)
        encoding = 2*torch.rand(1, self.encoding_size, device=self.device)
        return encoding, caption
 
if __name__ == '__main__':
    unittest.main()













  


