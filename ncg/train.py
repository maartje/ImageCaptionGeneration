from torch.utils import data
import torch
import torch.nn as nn
from torch import optim

from ncg.io.image_caption_dataset import ImageCaptionDataset
from ncg.nn.show_and_tell import ShowAndTell
from ncg.nn.train_model import train_iter
from ncg.reporting.loss_collector import LossCollector

def train(fpaths_image_encodings, fpaths_captions, vocab_size, encoding_size,
          learning_rate = 0.005, max_epochs = 50, dl_params = {}, store_loss_every = 100):
    # data loader
    dataset_train = ImageCaptionDataset(fpaths_image_encodings, fpaths_captions)
    dataloader_train = data.DataLoader(dataset_train, **dl_params)
    
    # model
#    im1 = torch.load(fpaths_image_encodings[0])
#    encoding_size = im1.size()[0]
    decoder = ShowAndTell(encoding_size, vocab_size)
    
    # optimization
    optimizer = optim.SGD(decoder.decoder.parameters(), lr = learning_rate)
    loss_criterion = nn.NLLLoss()    

    # loss collection
    loss_collector = LossCollector(store_loss_every)   
    assert store_loss_every < len(dataset_train)     
    
    
    def print_loss_info(e, i, l, epoch_finished):
        if epoch_finished:
            print (loss_collector.epoch_losses[-1])
        
    fns_on_update = [
        loss_collector.process_loss,
        print_loss_info
    ]
    train_iter(decoder, dataloader_train, loss_criterion, 
               optimizer, max_epochs, fns_on_update = fns_on_update)
    
    #torch.save(loss_collector, fpath_train_losses)
               
               
        
    


# store and print average epoch losses
# store and print average losses per i examples
# after epoch: calculate, store, print validation loss

# save train_epoch_losses
# save train_iter_losses
# save val_epoch_losses
# save model for each epoch



