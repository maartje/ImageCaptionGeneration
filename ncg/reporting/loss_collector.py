from datetime import datetime
import math
from ncg.debug_helpers import format_duration

class LossCollector():

    def __init__(self, epoch_size, batch_size, initial_validation_loss = None):
        self.initial_validation_loss = initial_validation_loss
        self.batch_size = batch_size
        self.epoch_losses_val = []
        self.epoch_losses_train = []
        self.batch_losses_train = []
        self.epoch_size = epoch_size
        self.batch_size_last = epoch_size % batch_size
        if not self.batch_size_last: # no partial batch
            self.batch_size_last = self.batch_size  
    
    #fn_batch_listener
    def on_batch_completed(self, epoch, batch_index, batch_size, loss):
        if epoch > (len(self.batch_losses_train) - 1):
            self.batch_losses_train.append([])
        self.batch_losses_train[-1].append(loss)

    #fn_epoch_listener
    def on_epoch_completed(self, epoch, batch_index, validation_loss):
        if validation_loss:
            self.epoch_losses_val.append(validation_loss)
        batch_losses = self.batch_losses_train[-1]
        num = self.batch_size * (sum(batch_losses) - batch_losses[-1]) + self.batch_size_last * batch_losses[-1]
        den = self.batch_size * (len(batch_losses) - 1) + self.batch_size_last
        average_epoch_loss = num / den
        self.epoch_losses_train.append(average_epoch_loss)
    
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
                current_batch_size = self.batch_size if last_batch else self.batch_size_last
                total_prev = epoch_index * self.epoch_size + batch_index * self.batch_size
                total = total_prev + current_batch_size
                batch_intervals.append(total)
        batch_losses_flat = [l for bl in self.batch_losses_train for l in bl]
        return batch_intervals, batch_losses_flat

class LossReporter:

    def __init__(self, loss_collector, print_loss_every, start_time):
        self.loss_collector = loss_collector
        self.print_loss_every = math.ceil(print_loss_every / loss_collector.batch_size)
        self.start_time = start_time
    
    def report_initial_validation_loss(self):
        val_loss = 'UNKNOWN'
        if self.loss_collector.initial_validation_loss:
            val_loss = f'{self.loss_collector.initial_validation_loss:0.2}'  
        str_duration = format_duration(self.start_time, datetime.now())
        print(f'({str_duration})\t{0}\tUNKNOWN\t{val_loss} ')
    
    
    def on_batch_completed(self, epoch, batch_index, batch_size, loss):
        if (batch_index + 1) % self.print_loss_every == 0:
            print('    epoch', epoch, 'batch_index', batch_index, 'batch_size', batch_size,
            '#examples', (batch_index + 1) * batch_size, 'batch_loss', f'{loss:0.2}')

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
        plt.plot(intervals, losses, color='blue', label='train loss over batch')
        plt.xlabel('#training pairs')
        plt.ylabel('average token loss')
        plt.legend()
        if fname:
            _ = plt.savefig(fname)
        else:
            plt.show()

