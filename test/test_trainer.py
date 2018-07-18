"""
Tests the training of the models
"""

import unittest
import torch
import torch.nn as nn
from torch import optim

from ncg.nn.models import DecoderRNN, ShowTell
from test.test_helpers import generate_random_training_pair
from torch.utils import data
from ncg.io.image_features_description_dataset import collate_image_features_descriptions
from test.test_helpers import MockEmbeddingDescriptionDataset

from ncg.nn.trainer import Trainer
from ncg.nn.predict import predict

class TestTrain(unittest.TestCase):
       
    def test_train_iter_decoderRNN(self):
        decoder = DecoderRNN(
            MockEmbeddingDescriptionDataset.encoding_size, 
            MockEmbeddingDescriptionDataset.vocab_size,
            -1)
        self.check_train_iter(decoder, 0.3, do_predict = False)

    def test_train_iter_ShowTell(self):
        hidden_size = 60
        show_tell = ShowTell(
            MockEmbeddingDescriptionDataset.encoding_size, 
            hidden_size, 
            MockEmbeddingDescriptionDataset.vocab_size, -1)
        self.check_train_iter(show_tell, 0.2)
        
    def check_train_iter(self, decoder, lr = 0.25, do_predict = True):
        loss_criterion = nn.NLLLoss()
        optimizer = optim.SGD(decoder.parameters(), lr = lr)
        ds = MockEmbeddingDescriptionDataset(range(3), range(1))
        collate = lambda b: collate_image_features_descriptions(b, 1000)
        dl_params = {
            'batch_size' : 1, 
            'collate_fn' : collate
        }
        train_data = data.DataLoader(ds, **dl_params)
        losses = []
        
        max_epochs = 10
        def stop_criterion(epoch, val_loss):
            return epoch > max_epochs
        
        trainer = Trainer(decoder, loss_criterion, optimizer)

        trainer.train_iter(train_data, stop_criterion, 
                   fn_batch_listeners = [lambda e,i,s,l: losses.append(l)])        
        losses_1 = losses[::3] # losses for pair 1
        losses_2 = losses[1::3]
        losses_3 = losses[2::3]
        
        #print()
        #print(losses_1)
        #print(losses_2)
        #print(losses_3)
        
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(is_decreasing(losses_1))
        self.assertTrue(is_decreasing(losses_2))
        self.assertTrue(is_decreasing(losses_3))
        
        if do_predict:
            # Make sure to overfit by doing another training cycle
            max_epochs = 15
            trainer.train_iter(train_data, stop_criterion, 
                       fn_batch_listeners = [lambda e,i,s,l: losses.append(l)])        


            # Assert that predicted and target look similar for (overfitted) train data
            dl_predict_params = {
                'batch_size' : 3
            }
            ds_images = [e for e, t in ds]
            targets = [t.numpy() for e, t in ds]
            
            predict_data = data.DataLoader(ds_images, **dl_predict_params)
            predicted_indices = predict(decoder, predict_data, 0, 20)
            for i in range(len(targets)):
                predicted = predicted_indices[i]
                target = targets[i]
                intersection = set(predicted) & set(target)

                #print()
                #print(predicted, 'predicted')
                #print(target, 'target')
                #print(intersection, 'intersection')

                self.assertTrue(len(intersection) > 0.5*len(target))
             
if __name__ == '__main__':
    unittest.main()













  


