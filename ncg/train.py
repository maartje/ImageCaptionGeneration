from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime, timedelta

from ncg.io.image_features_description_dataset import ImageFeaturesDescriptionsDataset, collate_image_features_descriptions

from ncg.nn.models import ShowTell
from ncg.nn.trainer import Trainer
from ncg.reporting.loss_collector import LossCollector, LossReporter
from ncg.debug_helpers import format_duration

def train(fpath_imfeats_train, fpaths_captions_train,
          fpath_imfeats_val, fpaths_captions_val, 
          hidden_size, vocab_size, PAD_index,
          fpath_loss_data_out, fpath_decoder_out,
          learning_rate = 0.005, max_epochs = 50, max_hours = 72, 
          dl_params_train = {}, dl_params_val = {}, 
          print_loss_every = 1000):


    # data loaders
    collate_fn = lambda b: collate_image_features_descriptions(b, PAD_index)
    dl_params_train['collate_fn'] = collate_fn
    dl_params_val['collate_fn'] = collate_fn
    dataset_train = ImageFeaturesDescriptionsDataset(fpath_imfeats_train, fpaths_captions_train)
    dataloader_train = data.DataLoader(dataset_train, **dl_params_train)
    dataset_val = ImageFeaturesDescriptionsDataset(fpath_imfeats_val, fpaths_captions_val)
    dataloader_val = data.DataLoader(dataset_val, **dl_params_val)
    
    # model
    encoding_size = dataset_val[0][0].size()[0]
    decoder = ShowTell(encoding_size, hidden_size, vocab_size, PAD_index)
    
    # optimization
    optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    loss_criterion = nn.NLLLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        factor = 0.5, 
        patience = 2, 
        verbose=True
    )
    # trainer    
    trainer = Trainer(decoder, loss_criterion, optimizer, scheduler)

    # loss collection
    loss_collector = LossCollector(len(dataset_train), dl_params_train['batch_size'], dataloader_val)  
    start_time = datetime.now()
    end_time = start_time + timedelta(hours = max_hours)
    loss_reporter = LossReporter(loss_collector, print_loss_every, start_time) 
    
    # calculate and store initial validation loss 
    print('\ntime passed', '  epoch', 'train_loss', 'val_loss')
    initial_val_loss = trainer.calculate_validation_loss(dataloader_val)
    loss_collector.initial_validation_loss = initial_val_loss
    loss_reporter.report_initial_validation_loss()      

    def on_epoch_completed(epoch, trainer):
        # save model and loss data
        torch.save(loss_collector.to_dict(), fpath_loss_data_out)
        torch.save(decoder, fpath_decoder_out)
    
    def stop_criterion(epoch):
        if datetime.now() > end_time:
            print(f'exceeded max hours {max_hours}')
            return True
        return epoch > max_epochs
        
    # train model and 
    # collect validation loss data
    # TODO: store model per X iterations?
    trainer.train_iter(dataloader_train, stop_criterion,
               fn_batch_listeners = [
                   loss_collector.on_batch_completed, loss_reporter.on_batch_completed],
               fn_epoch_listeners = [
                   loss_collector.on_epoch_completed, loss_reporter.on_epoch_completed, 
                   on_epoch_completed]
               )
               
    

