class LossCollector():

    def __init__(self, batch_loss_size):
        self.batch_loss_size = batch_loss_size
        self.epoch_losses = []
        self.batch_losses = []
        self._tmp_losses = []
        # TODO: epoch size, batch_sizes (for diagrams)
        
    def process_loss(self, epoch, batch_index, token_loss, epoch_finished):
        self._tmp_losses.append(token_loss)
        batch_mod = len(self._tmp_losses) % self.batch_loss_size
        if batch_mod == 0 and len(self._tmp_losses):
            if epoch >= len(self.batch_losses):
                self.batch_losses.append([])
            losses_current_batch = self._tmp_losses[-self.batch_loss_size:]
#            print('store batch', epoch, batch_index, losses_current_batch)
            average_batch_loss = sum(losses_current_batch)/len(losses_current_batch)
            self.batch_losses[epoch].append(average_batch_loss)
        if epoch_finished:        
            average_epoch_loss = sum(self._tmp_losses)/ len(self._tmp_losses)
            self.epoch_losses.append(average_epoch_loss)
#            print('store epoch', epoch, batch_index, losses[0])
            
            if batch_mod:
                losses_partial_batch = self._tmp_losses[-batch_mod:]
                average_partial_batch_loss = sum(losses_partial_batch)/len(losses_partial_batch)
                self.batch_losses[-1].append(average_partial_batch_loss)
#                print('store partial batch', epoch, batch_index, losses_partial_batch)
            
            self._tmp_losses = []


