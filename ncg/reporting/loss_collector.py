from datetime import datetime

from ncg.debug_helpers import format_duration

class LossCollector():

    def __init__(self, batch_loss_size, initial_validation_loss = None):
        self.initial_validation_loss = initial_validation_loss
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
    def on_epoch_completed(self, epoch, batch_index, validation_loss):
        self.epoch_losses_val.append(validation_loss)

        average_epoch_loss = sum(self._tmp_losses_train) / len(self._tmp_losses_train)
        self.epoch_losses_train.append(average_epoch_loss)
        self.epoch_size = batch_index + 1
#       print('store epoch', epoch, batch_index, losses[0])

        batch_mod = len(self._tmp_losses_train) % self.batch_loss_size
        self.batch_loss_size_last = self.batch_loss_size
        if batch_mod > 0:
            self.batch_loss_size_last = batch_mod
            self.store_batch_loss(batch_mod, epoch)
        self._tmp_losses_train = []

            
    def store_batch_loss(self, batch_size, epoch):
        if epoch > (len(self.batch_losses_train) - 1):
            self.batch_losses_train.append([])
        batch_losses = self._tmp_losses_train[-batch_size : ]
        average_batch_loss = sum(batch_losses)/len(batch_losses)
        self.batch_losses_train[-1].append(average_batch_loss)
#        print('store partial batch', epoch, batch_index, losses_partial_batch)
    
    def get_epoch_intervals(self, epoch_losses):
        return [(i + 1)*self.epoch_size for i in range(len(epoch_losses))] 

    def plot_values_epoch_losses_val(self):
        epoch_intervals = self.get_epoch_intervals(self.epoch_losses_val)
        if not self.initial_validation_loss:
            return epoch_intervals, self.epoch_losses_val
        epoch_intervals = [0] + epoch_intervals
        epoch_losses_val = [self.initial_validation_loss] + self.epoch_losses_val
        return epoch_intervals, epoch_losses_val
        
    def plot_values_epoch_losses_train(self):
        epoch_intervals = self.get_epoch_intervals(self.epoch_losses_train)
        return epoch_intervals, self.epoch_losses_train

    def plot_values_batch_losses_train(self):
        batch_intervals = []
        for epoch_index, bl in enumerate(self.batch_losses_train):
            for batch_index, _ in enumerate(bl):
                last_batch = batch_index < (len(bl) - 1)
                current_batch_size = self.batch_loss_size if last_batch else self.batch_loss_size_last
                total_prev = epoch_index * self.epoch_size + batch_index * self.batch_loss_size
                total = total_prev + current_batch_size
                batch_intervals.append(total)
        batch_losses_flat = [l for bl in self.batch_losses_train for l in bl]
        return batch_intervals, batch_losses_flat

class LossReporter:

    def __init__(self, loss_collector, print_loss_every, start_time):
        self.loss_collector = loss_collector
        self.print_loss_every = print_loss_every
        self.start_time = start_time
    
    def report_initial_validation_loss(self):
        val_loss = 'UNKNOWN'
        if self.loss_collector.initial_validation_loss:
            val_loss = f'{self.loss_collector.initial_validation_loss:0.2}'  
        str_duration = format_duration(self.start_time, datetime.now())
        print(f'({str_duration})\t{0}\tUNKNOWN\t{val_loss} ')
    
    
    def on_batch_completed(self, epoch, batch_index, token_loss):
        if (batch_index + 1) % self.print_loss_every == 0:
            print('    epoch', epoch, 'batch_index', batch_index, 'instance_loss', f'{token_loss:0.2}')

    def on_epoch_completed(self, epoch, batch_index, val_loss):
        val_loss = 'UNKNOWN'
        if self.loss_collector.epoch_losses_val:
            val_loss = f'{self.loss_collector.epoch_losses_val[-1]:0.2}'  
        train_loss = self.loss_collector.epoch_losses_train[-1]
        str_duration = format_duration(self.start_time, datetime.now())
        print(f'({str_duration})\t{epoch + 1}\t{train_loss:0.2}\t{val_loss} ')


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class LossPlotter():

    def __init__(self, loss_collector):
        self.loss_collector = loss_collector

    def plotEpochLosses(self, fname = None):
        intervals_val, losses_val = self.loss_collector.plot_values_epoch_losses_val()
        intervals_train, losses_train = self.loss_collector.plot_values_epoch_losses_train()
        
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.ticklabel_format(axis='x', scilimits=(0, 0))
        if losses_train:
            plt.plot(intervals_train, losses_train, 'ro-', color='blue', label='train loss over epoch')
        if losses_val:
            plt.plot(intervals_val, losses_val, 'ro-', color='red', label='validation loss after epoch')
        plt.xlabel('#training pairs')
        plt.ylabel('average token loss')
        plt.legend()
        if fname:
            _ = plt.savefig(fname)
        else:
            plt.show()

    # TODO: extract building the frame
    def plotBatchLosses(self, fname = None):
        intervals, losses = self.loss_collector.plot_values_batch_losses_train()        
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.ticklabel_format(axis='x', scilimits=(0, 0))
        plt.plot(intervals, losses, 'ro-', color='blue', label='train loss over batch')
        plt.xlabel('#training pairs')
        plt.ylabel('average token loss')
        plt.legend()
        if fname:
            _ = plt.savefig(fname)
        else:
            plt.show()

