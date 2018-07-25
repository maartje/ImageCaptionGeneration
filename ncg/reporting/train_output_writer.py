from ncg.debug_helpers import format_duration
from datetime import datetime
import math

class TrainOutputWriter:

    def __init__(self, loss_collector, bleu_collector, print_loss_every, start_time):
        self.loss_collector = loss_collector
        self.bleu_collector = bleu_collector
        self.print_loss_every = math.ceil(print_loss_every / loss_collector.batch_size)
        self.start_time = start_time
        
    def on_batch_completed(self, epoch, batch_index, batch_size, loss):
        if (batch_index + 1) % self.print_loss_every == 0:
            print('    epoch', epoch, 'batch_index', batch_index, 'batch_size', batch_size,
            '#examples', (batch_index + 1) * batch_size, 'batch_loss', f'{loss:0.2}')

    def on_epoch_completed(self, epoch, trainer):
        if epoch == -1:
            print('\ntime passed', '  epoch', 'train_loss', 'val_loss', 'val_bleu')
        val_loss = 'UNKNOWN'
        if self.loss_collector.epoch_losses_val:
            val_loss = f'{self.loss_collector.epoch_losses_val[-1]:0.2}' 
        bleu = 'UNKNOWN'
        if self.bleu_collector.bleu_val: 
            val_bleu = f'{self.bleu_collector.bleu_val[-1]:0.2}' 
        train_loss_str = 'UNKNOWN'
        if epoch > -1:            
            train_loss = self.loss_collector.epoch_losses_train[-1]
            train_loss_str = f'{train_loss:0.2}'
        str_duration = format_duration(self.start_time, datetime.now())
        print(f'({str_duration})\t{epoch + 1}\t{train_loss_str}\t{val_loss}\t{val_bleu}')

