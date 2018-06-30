from datetime import datetime

from ncg.debug_helpers import format_duration

class LossCollector():

    def __init__(self, batch_loss_size):
        self.batch_loss_size = batch_loss_size
        self.epoch_losses_val = []
        self.epoch_losses_train = []
        self.batch_losses_train = []
        self._tmp_losses_train = []
        self.epoch_size = None # determined during processing
        self.batch_loss_size_last = 0 # determined during processing
    
    #fn_batch_listener
    def on_batch_completed(self, epoch, batch_index, token_loss):
        self._tmp_losses_train.append(token_loss)
        batch_mod = len(self._tmp_losses_train) % self.batch_loss_size
        if batch_mod == 0:
            self.store_batch_loss(self.batch_loss_size, epoch)

    #fn_epoch_listener
    def on_epoch_completed(self, epoch, batch_index, val_loss):
        average_epoch_loss = sum(self._tmp_losses_train) / len(self._tmp_losses_train)
        self.epoch_losses_train.append(average_epoch_loss)
        self.epoch_size = batch_index + 1
#       print('store epoch', epoch, batch_index, losses[0])

        batch_mod = len(self._tmp_losses_train) % self.batch_loss_size
        self.batch_loss_size_last = batch_mod
        if batch_mod > 0:
            self.store_batch_loss(batch_mod, epoch)
        self._tmp_losses_train = []

            
    def store_batch_loss(self, batch_size, epoch):
        if epoch > (len(self.batch_losses_train) - 1):
            self.batch_losses_train.append([])
        batch_losses = self._tmp_losses_train[-batch_size : ]
        average_batch_loss = sum(batch_losses)/len(batch_losses)
        self.batch_losses_train[-1].append(average_batch_loss)
#        print('store partial batch', epoch, batch_index, losses_partial_batch)
            
    def update_validation_loss(self, validation_loss):
        self.epoch_losses_val.append(validation_loss)
        
    def print_loss_info(self, epoch, batch_index, token_loss, epoch_finished):
        if epoch == 0 and batch_index == 0 and len(self.epoch_losses_val):
            val_loss = self.epoch_losses_val[-1]
            str_duration = format_duration(self.start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\ttrain_loss: __ \t{val_loss:0.2} ')            
        if epoch_finished:
            val_loss = self.epoch_losses_val[-1]
            train_loss = self.epoch_losses_train[-1]
            str_duration = format_duration(self.start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\t{train_loss:0.2}\t{val_loss:0.2} ')

class LossReporter:

    def __init__(self, loss_collector, print_loss_every, start_time):
        self.loss_collector = loss_collector
        self.print_loss_every = print_loss_every
        self.start_time = start_time
    
    def on_batch_completed(self, epoch, batch_index, token_loss):
        if (batch_index + 1) % self.print_loss_every == 0:
            print('    epoch', epoch, 'batch_index', batch_index, 'instance_loss', f'{token_loss:0.2}')

    def on_epoch_completed(self, epoch, batch_index, val_loss):
        if self.loss_collector.epoch_losses_val:
            val_loss = f'{self.loss_collector.epoch_losses_val[-1]:0.2}'  
        else: 
            val_loss = 'UNKNOWN'
        train_loss = self.loss_collector.epoch_losses_train[-1]
        str_duration = format_duration(self.start_time, datetime.now())
        print(f'({str_duration})\t{epoch + 1}\t{train_loss:0.2}\t{val_loss} ')

