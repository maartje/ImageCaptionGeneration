from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

from ncg.io.image_features_description_dataset import ImageFeaturesDescriptionsDataset, collate_image_features_descriptions
from ncg.io.image_features_dataset import ImageFeaturesDataset

from ncg.nn.models import ShowTell
from ncg.nn.trainer import Trainer
from ncg.reporting.loss_collector import LossCollector
from ncg.reporting.bleu_collector import BleuCollector
from ncg.reporting.train_output_writer import TrainOutputWriter
from ncg.debug_helpers import format_duration
from ncg.model_saver import ModelSaver

def train(fpath_imfeats_train, fpaths_captions_train,
          fpath_imfeats_val, fpaths_captions_val, 
          hidden_size, fpath_vocab, max_length,
          fpath_loss_data_out, fpath_bleu_scores_out, fpath_model, fpath_model_best, 
          learning_rate = 0.005, max_epochs = 50, max_hours = 72, 
          dl_params_train = {}, dl_params_val = {}, clip = 5,
          print_loss_every = 1000):

    # text info
    text_mapper = torch.load(fpath_vocab)
    vocab_size = text_mapper.vocab.n_words
    PAD_index = text_mapper.PAD_index()
    
    # data loaders
    dataset_imfeats_val = ImageFeaturesDataset(fpath_imfeats_val)
    dl_image_features_val = data.DataLoader(dataset_imfeats_val, **dl_params_val)
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
    loss_criterion = nn.NLLLoss(ignore_index = PAD_index)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        factor = 0.5, 
        patience = 3, 
        verbose=True,
        cooldown = 5,
        min_lr = 0.001
    )
    if clip:
        clip_grad_norm_(decoder.parameters(), clip)
    
    # trainer    
    trainer = Trainer(decoder, loss_criterion, optimizer, scheduler)

    # loss collection
    loss_collector = LossCollector(
        len(dataset_train), dl_params_train['batch_size'], fpath_loss_data_out, dataloader_val)  
    start_time = datetime.now()
    end_time = start_time + timedelta(hours = max_hours)
    
    # bleu collection
    references = [torch.load(fpath) for fpath in fpaths_captions_val]
    bleu_collector = BleuCollector(
        dl_image_features_val, references, text_mapper, max_length, 
        len(dataset_train), fpath_bleu_scores_out)

    # reporting
    output_writer = TrainOutputWriter(loss_collector, bleu_collector, print_loss_every, start_time) 
    
    # save intermediate results
    model_saver = ModelSaver(decoder, bleu_collector, fpath_model, fpath_model_best)
        
    # stopper
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
                   loss_collector.on_batch_completed, output_writer.on_batch_completed],
               fn_epoch_listeners = [
                   loss_collector.on_epoch_completed, bleu_collector.on_epoch_completed, 
                   output_writer.on_epoch_completed, model_saver.on_epoch_completed]
               )
               
    

