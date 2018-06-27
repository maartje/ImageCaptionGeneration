from torch.utils import data
import torch
import torch.nn as nn
from torch import optim

from ncg.io.image_caption_dataset import ImageCaptionDataset
from ncg.nn.show_and_tell import ShowAndTell
from ncg.nn.train_model import train_iter

def train(fpaths_image_encodings, fpaths_captions, vocab_size, encoding_size,
          learning_rate = 0.005, max_epochs = 5, dl_params = {}):
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

    train_iter(decoder, dataloader_train, loss_criterion, 
               optimizer, max_epochs, fn_on_update = store_train_loss)
               
def store_train_loss(epoch, i, token_loss):
    print(epoch, i, token_loss)
