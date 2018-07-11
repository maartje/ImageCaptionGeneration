from torch.utils import data
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime, timedelta

from ncg.io.embedding_description_dataset import EmbeddingDescriptionGroupedDataset
from ncg.nn.models import DecoderRNN
from ncg.nn.train_model import train_iter, calculate_validation_loss
from ncg.reporting.loss_collector import LossCollector, LossReporter
from ncg.debug_helpers import format_duration

def train(fpaths_images_train, fpaths_captions_train,
          fpaths_images_val, fpaths_captions_val, 
          vocab_size, encoding_size,
          fpath_loss_data_out, fpath_decoder_out,
          learning_rate = 0.005, max_epochs = 50, max_hours = 72, dl_params = {}, 
          store_loss_every = 100, print_loss_every = 1000):

    # data loaders
    dataset_train = EmbeddingDescriptionGroupedDataset(fpaths_images_train, fpaths_captions_train)
    dataloader_train = data.DataLoader(dataset_train, **dl_params)
    dataset_val = EmbeddingDescriptionGroupedDataset(fpaths_images_val, fpaths_captions_val)
    dataloader_val = data.DataLoader(dataset_val, **dl_params)
    
    # model
    decoder = DecoderRNN(encoding_size, vocab_size)
    
    # optimization
    optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    loss_criterion = nn.NLLLoss()    

    # loss collection
    assert store_loss_every < len(dataset_train)     
    loss_collector = LossCollector(store_loss_every)  
    start_time = datetime.now()
    end_time = start_time + timedelta(hours = max_hours)
    loss_reporter = LossReporter(loss_collector, print_loss_every, start_time) 
    
    # calculate and store initial validation loss 
    print('\ntime passed', '  epoch', 'train_loss', 'val_loss')
    initial_val_loss = calculate_validation_loss(decoder, dataloader_val, loss_criterion)
    loss_collector.initial_validation_loss = initial_val_loss
    loss_reporter.report_initial_validation_loss()      

    def on_epoch_completed(epoch, batch_index, validation_loss):
        # save model and loss data
        torch.save(loss_collector, fpath_loss_data_out)
        torch.save(decoder, fpath_decoder_out)
    
    def stop_criterion(epoch, val_loss):
        if datetime.now() > end_time:
            print(f'exceeded max hours {max_hours}')
            return True
        return epoch > max_epochs
        
    # train model and 
    # collect validation loss data
    # TODO: store model per X iterations?
    train_iter(decoder, dataloader_train, loss_criterion, 
               optimizer, stop_criterion, 
               val_data = dataloader_val,
               fn_batch_listeners = [
                   loss_collector.on_batch_completed, loss_reporter.on_batch_completed],
               fn_epoch_listeners = [
                   loss_collector.on_epoch_completed, loss_reporter.on_epoch_completed, 
                   on_epoch_completed]
               )
               
    

