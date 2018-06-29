from torch.utils import data
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime

from ncg.io.image_caption_dataset import ImageCaptionDataset
from ncg.nn.show_and_tell import ShowAndTell
from ncg.nn.train_model import train_iter, calculate_validation_loss
from ncg.reporting.loss_collector import LossCollector
from ncg.debug_helpers import format_duration

def train(fpaths_images_train, fpaths_captions_train,
          fpaths_images_val, fpaths_captions_val, 
          vocab_size, encoding_size,
          fpath_loss_data_out, fpath_decoder_out,
          max_train_instances = None, #TODO pass into dataset
          learning_rate = 0.005, max_epochs = 50, dl_params = {}, 
          store_loss_every = 100, print_loss_every = 1000):
    # data loader
    dataset_train = ImageCaptionDataset(fpaths_images_train, fpaths_captions_train)
    dataloader_train = data.DataLoader(dataset_train, **dl_params)
    dataset_val = ImageCaptionDataset(fpaths_images_val, fpaths_captions_val)
    dataloader_val = data.DataLoader(dataset_val, **dl_params)
    
    # model
#    im1 = torch.load(fpaths_images_train[0])
#    encoding_size = im1.size()[0]
    decoder = ShowAndTell(encoding_size, vocab_size)
    
    # optimization
    optimizer = optim.SGD(decoder.decoder.parameters(), lr = learning_rate)
    loss_criterion = nn.NLLLoss()    

    # loss collection
    loss_collector = LossCollector(store_loss_every)   
    assert store_loss_every < len(dataset_train)     
    
    epoch_val_losses = []
    def collect_validation_loss(e, i, l, epoch_finished):
        if epoch_finished or (e == 0 and i == 0):
            val_loss = calculate_validation_loss(decoder, dataloader_val, loss_criterion)
            epoch_val_losses.append(val_loss)
    
    def print_loss_info(epoch, i, l, epoch_finished):
        if (i + 1) % print_loss_every == 0:
            print('epoch', epoch, 'i', i, 'instance_loss', f'{l:0.2}')
        if epoch == 0 and i == 0:
            val_loss = epoch_val_losses[-1]
            str_duration = format_duration(start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\ttrain_loss: __ \t{val_loss:0.2} ')            
        if epoch_finished:
            val_loss = epoch_val_losses[-1]
            train_loss = loss_collector.epoch_losses[-1]
            
            str_duration = format_duration(start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\t{train_loss:0.2}\t{val_loss:0.2} ')
        
    fns_on_update = [
        loss_collector.update_train_loss,
        collect_validation_loss,
        print_loss_info
    ]
    
    start_time = datetime.now()
    print('\ntime passed', '  epoch', 'train_loss', 'val_loss')
    train_iter(decoder, dataloader_train, loss_criterion, 
               optimizer, max_epochs, fns_on_update = fns_on_update)
    
    loss_data = {
        'epoch_val_losses' : epoch_val_losses,
        'epoch_train_losses' : loss_collector.epoch_losses,
        'batch_losses' : loss_collector.batch_losses,
        'batch_loss_size' : loss_collector.batch_loss_size,
        'epoch_size' : len(dataloader_train) 
    }
    torch.save(loss_data, fpath_loss_data_out)
    torch.save(decoder, fpath_decoder_out)
    
    
    
#    for source_encodings, targets in dataloader_train:
#        source_encodings, targets = source_encodings.to(device), targets.to(device)
#        loss = calculate_loss(decoder, source_encodings, targets, loss_criterion)
#        print('loss', loss)
 #       predicted = decoder.predict(train_input[0], 0, 1, 20, device)
 #       print('predicted       ', predicted)
 #       print('target', train_input[1])
 #       print()
               
               
        
    


# store and print average epoch losses
# store and print average losses per i examples
# after epoch: calculate, store, print validation loss

# save train_epoch_losses
# save train_iter_losses
# save val_epoch_losses
# save model for each epoch



