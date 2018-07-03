"""
Tests the training of the models
"""

import unittest
import torch
import torch.nn as nn
from torch import optim

from ncg.nn.train_model import train_iter, predict
from ncg.nn.models import DecoderRNN
from test.test_helpers import generate_random_training_pair


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.encoding_size = 256
        self.loss_criterion = nn.NLLLoss()
       
    def test_train_iter_decoderRNN(self):
        decoder = DecoderRNN(self.encoding_size, self.vocab_size)
        optimizer = optim.SGD(decoder.parameters(), lr = 0.2)

        train_data = [
            generate_random_training_pair(self.encoding_size, self.vocab_size, 10),
            generate_random_training_pair(self.encoding_size, self.vocab_size, 6),
            generate_random_training_pair(self.encoding_size, self.vocab_size, 9),
        ]
        losses = []
        train_iter(decoder, train_data, self.loss_criterion, optimizer, max_epochs = 10, 
                   fn_batch_listeners = [lambda e,i,l: losses.append(l)])        
        losses_1 = losses[::3] # losses for pair 1
        losses_2 = losses[1::3]
        losses_3 = losses[2::3]
        
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(is_decreasing(losses_1))
        self.assertTrue(is_decreasing(losses_2))
        self.assertTrue(is_decreasing(losses_3))
        
        # Assert that predicted and target look similar for (overfitted) train data
        for train_instance in train_data:
            predicted = predict(decoder, train_instance[0], 0, 1, 20)
            target = train_instance[1].squeeze().numpy()
            intersection = set(predicted) & set(target)
            self.assertTrue(len(intersection) > 0.5*len(target))
 
if __name__ == '__main__':
    unittest.main()













  


