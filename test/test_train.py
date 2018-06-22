"""
Tests the training of the models
"""

import unittest
import torch
import torch.nn as nn
from torch import optim

from ncg.models.train import train_iter
from ncg.models.show_and_tell import ShowAndTell

def generate_random_training_pair(vocab_size, encoding_size, device):
    caption = torch.LongTensor(10,1, device=device).random_(0, vocab_size)
    encoding = 2*torch.rand(encoding_size, device=device)
    return encoding, caption

def generate_random_training_batches(train_size, batch_size, 
                                     vocab_size, encoding_size, device):
    training_pairs = [
       generate_random_training_pair(vocab_size, encoding_size, device) for _ in range(train_size)]
    steps = range(0, train_size, batch_size)
    batches = [training_pairs[i:i + batch_size] for i in steps if i + batch_size <= train_size]
    return batches

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.encoding_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_criterion = nn.NLLLoss()
       
    def test_train_iter_show_and_tell(self):
        decoder = ShowAndTell(self.encoding_size, self.vocab_size, self.device)
        optimizer = optim.SGD(decoder.decoder.parameters(), lr = 1.0)

        train_size = 8
        batch_size = 4
        batches = generate_random_training_batches(
            train_size, batch_size, self.vocab_size, self.encoding_size, self.device)

        batch_losses = train_iter(decoder, batches, self.loss_criterion, optimizer, epochs = 5)
        batch_1_losses = batch_losses[::2]
        batch_2_losses = batch_losses[1::2]
        
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(is_decreasing(batch_1_losses))
        self.assertTrue(is_decreasing(batch_2_losses))
 
if __name__ == '__main__':
    unittest.main()













  


