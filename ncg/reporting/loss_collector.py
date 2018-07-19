def createLossCollectorFromDict(state_dict):
    lossCollector = LossCollector(state_dict['epoch_size'], state_dict['batch_size'])
    lossCollector.initial_validation_loss = state_dict['initial_validation_loss']
    lossCollector.epoch_losses_val = state_dict['epoch_losses_val']
    lossCollector.epoch_losses_train = state_dict['epoch_losses_train']
    lossCollector.batch_losses_train = state_dict['batch_losses_train']
    lossCollector.batch_size_last = state_dict['batch_size_last']
    return lossCollector

class LossCollector():

    def __init__(self, epoch_size, batch_size, val_data = None, initial_validation_loss = None):
        self.initial_validation_loss = initial_validation_loss
        self.batch_size = batch_size
        self.epoch_losses_val = []
        self.epoch_losses_train = []
        self.batch_losses_train = []
        self.epoch_size = epoch_size
        self.batch_size_last = epoch_size % batch_size
        if not self.batch_size_last: # no partial batch
            self.batch_size_last = self.batch_size  
        self.val_data = val_data
    
    def to_dict(self):
        return {
            'initial_validation_loss' : self.initial_validation_loss,
            'batch_size' : self.batch_size,
            'epoch_losses_val' : self.epoch_losses_val,
            'epoch_losses_train' : self.epoch_losses_train,
            'batch_losses_train' : self.batch_losses_train,
            'epoch_size' : self.epoch_size,
            'batch_size_last' : self.batch_size_last
        }
        
        
    
    #fn_batch_listener
    def on_batch_completed(self, epoch, batch_index, batch_size, loss):
        if epoch > (len(self.batch_losses_train) - 1):
            self.batch_losses_train.append([])
        self.batch_losses_train[-1].append(loss)

    #fn_epoch_listener
    def on_epoch_completed(self, epoch, trainer):
        if self.val_data:
            validation_loss = trainer.calculate_validation_loss(self.val_data)
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

