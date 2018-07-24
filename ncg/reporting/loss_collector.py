import torch

class LossCollector():

    def __init__(self, epoch_size, batch_size, fpath_loss_data, val_data = None):
        self.batch_size = batch_size
        self.epoch_losses_val = []
        self.epoch_losses_train = []
        self.batch_losses_train = []
        self.epoch_size = epoch_size
        self.batch_size_last = epoch_size % batch_size
        if not self.batch_size_last: # no partial batch
            self.batch_size_last = self.batch_size  
        self.val_data = val_data
        self.fpath_loss_data = fpath_loss_data
        
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
        if self.batch_losses_train:
            batch_losses = self.batch_losses_train[-1]
            num = self.batch_size * (sum(batch_losses) - batch_losses[-1]) + self.batch_size_last * batch_losses[-1]
            den = self.batch_size * (len(batch_losses) - 1) + self.batch_size_last
            average_epoch_loss = num / den
            self.epoch_losses_train.append(average_epoch_loss)
        torch.save(self.plot_values(), self.fpath_loss_data)
    
    def plot_values_epoch_losses_val(self):
        epoch_intervals = [i*self.epoch_size for i in range(len(self.epoch_losses_val))]
        return epoch_intervals, self.epoch_losses_val
        
    def plot_values_epoch_losses_train(self):
        epoch_intervals = [(i + 1)*self.epoch_size for i in range(len(self.epoch_losses_train))]
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

    def plot_values(self):
        (batch_intervals, batch_losses) = self.plot_values_batch_losses_train()
        (epoch_intervals_train, epoch_losses_train) = self.plot_values_epoch_losses_train()
        (epoch_intervals_val, epoch_losses_val) = self.plot_values_epoch_losses_val()
        return (batch_intervals, batch_losses, epoch_intervals_train, epoch_losses_train, epoch_intervals_val, epoch_losses_val)


