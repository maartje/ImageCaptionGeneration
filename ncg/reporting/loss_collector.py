class LossCollector():

    def __init__(self, batch_loss_size):
        self.batch_loss_size = batch_loss_size
        self.epoch_losses = []
        self.batch_losses = []
        self._tmp_losses = []
        self.epoch_size = None # determined during processing
        self.batch_loss_size_last = 0 # determined during processing
        
    def process_train_loss(self, epoch, batch_index, token_loss, epoch_finished):
        self.update_tmp_loss(epoch, batch_index, token_loss, epoch_finished)
        self.update_batch_loss(epoch, batch_index, token_loss, epoch_finished)
        self.update_epoch_loss(epoch, batch_index, token_loss, epoch_finished)

    # Step 1: add loss to _tmp_losses, clear _tmp_losses when new epoch has started
    def update_tmp_loss(self, epoch, batch_index, token_loss, epoch_finished):
        if batch_index == 0:        
            self._tmp_losses = []
        self._tmp_losses.append(token_loss)

    # Step 2: store batch loss in batch losses, when batch interval
    def update_batch_loss(self, epoch, batch_index, token_loss, epoch_finished):
        batch_mod = len(self._tmp_losses) % self.batch_loss_size
        if batch_mod == 0 and len(self._tmp_losses):
            if epoch >= len(self.batch_losses):
                self.batch_losses.append([])
            self.store_batch_loss(self.batch_loss_size)
        elif epoch_finished:
            self.batch_loss_size_last = batch_mod
            self.store_batch_loss(batch_mod)
            
    def store_batch_loss(self, batch_size):
        batch_losses = self._tmp_losses[-batch_size : ]
        average_batch_loss = sum(batch_losses)/len(batch_losses)
        self.batch_losses[-1].append(average_batch_loss)
#        print('store partial batch', epoch, batch_index, losses_partial_batch)
        
    # Step 3: store epoch loss in epoch_losses when epoch has finished
    def update_epoch_loss(self, epoch, batch_index, token_loss, epoch_finished):
        if epoch_finished:        
            average_epoch_loss = sum(self._tmp_losses)/ len(self._tmp_losses)
            self.epoch_losses.append(average_epoch_loss)
            self.epoch_size = batch_index + 1
#            print('store epoch', epoch, batch_index, losses[0])
            

