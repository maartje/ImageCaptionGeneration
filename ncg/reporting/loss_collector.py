from datetime import datetime

from ncg.debug_helpers import format_duration

class LossCollector():

    def __init__(self, batch_loss_size, print_loss_every, start_time):
        self.batch_loss_size = batch_loss_size
        self.print_loss_every = print_loss_every
        self.start_time = start_time
        self.epoch_losses_val = []
        self.epoch_losses_train = []
        self.batch_losses_train = []
        self._tmp_losses_train = []
        self.epoch_size = None # determined during processing
        self.batch_loss_size_last = 0 # determined during processing
        
    def update_train_loss(self, epoch, batch_index, token_loss, epoch_finished):
        self.update_tmp_loss(epoch, batch_index, token_loss, epoch_finished)
        self.update_batch_loss(epoch, batch_index, token_loss, epoch_finished)
        self.update_epoch_loss(epoch, batch_index, token_loss, epoch_finished)

    # Step 1: add loss to _tmp_losses, clear _tmp_losses when new epoch has started
    def update_tmp_loss(self, epoch, batch_index, token_loss, epoch_finished):
        if batch_index == 0:        
            self._tmp_losses_train = []
        self._tmp_losses_train.append(token_loss)

    # Step 2: store batch loss in batch losses, when batch interval
    def update_batch_loss(self, epoch, batch_index, token_loss, epoch_finished):
        batch_mod = len(self._tmp_losses_train) % self.batch_loss_size
        if batch_mod == 0 and len(self._tmp_losses_train):
            if epoch >= len(self.batch_losses_train):
                self.batch_losses_train.append([])
            self.store_batch_loss(self.batch_loss_size)
        elif epoch_finished:
            self.batch_loss_size_last = batch_mod
            self.store_batch_loss(batch_mod)
            
    def store_batch_loss(self, batch_size):
        batch_losses = self._tmp_losses_train[-batch_size : ]
        average_batch_loss = sum(batch_losses)/len(batch_losses)
        self.batch_losses_train[-1].append(average_batch_loss)
#        print('store partial batch', epoch, batch_index, losses_partial_batch)
        
    # Step 3: store epoch loss in epoch_losses when epoch has finished
    def update_epoch_loss(self, epoch, batch_index, token_loss, epoch_finished):
        if epoch_finished:        
            average_epoch_loss = sum(self._tmp_losses_train)/ len(self._tmp_losses_train)
            self.epoch_losses_train.append(average_epoch_loss)
            self.epoch_size = batch_index + 1
#            print('store epoch', epoch, batch_index, losses[0])
            
    def update_validation_loss(self, validation_loss):
        self.epoch_losses_val.append(validation_loss)
        
    def print_loss_info(self, epoch, batch_index, token_loss, epoch_finished):
        if (batch_index + 1) % self.print_loss_every == 0:
            print('    epoch', epoch, 'batch_index', batch_index, 'instance_loss', f'{token_loss:0.2}')
        if epoch == 0 and batch_index == 0 and len(self.epoch_losses_val):
            val_loss = self.epoch_losses_val[-1]
            str_duration = format_duration(self.start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\ttrain_loss: __ \t{val_loss:0.2} ')            
        if epoch_finished:
            val_loss = self.epoch_losses_val[-1]
            train_loss = self.epoch_losses_train[-1]
            str_duration = format_duration(self.start_time, datetime.now())
            print(f'({str_duration})\t{epoch + 1}\t{train_loss:0.2}\t{val_loss:0.2} ')


