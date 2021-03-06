import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim

class Trainer:

    def __init__(self, decoder, loss_criterion, optimizer_type, lr,
            lr_decay = [], alpha_c = 1.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = decoder
        self.loss_criterion = loss_criterion 

        # construct after move model to cuda
        self.scheduler = None
        self.optimizer = None 
        
        # used for configuring the optimizer
        self.optimizer_type = optimizer_type
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.alpha_c = alpha_c
        

    def train_iter(self, train_data, 
                   fn_stop_criterion,
                   fn_batch_listeners = [], fn_epoch_listeners = []):
        self.decoder.to(self.device)
        self.loss_criterion.to(self.device)
        self.set_optimizer()
        
        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(-1, self)
        epoch = 0
        while not fn_stop_criterion(epoch):
            self.decoder.train()
            total_train_loss = 0
            for batch_index, batch in enumerate(train_data):
                token_loss = self.train(batch)
                total_train_loss += token_loss
                for fn_on_batch_completed in fn_batch_listeners:
                    batch_size = batch[0].size()[0] # may differ for last batch
                    fn_on_batch_completed(epoch, batch_index, batch_size, token_loss)
            for fn_on_epoch_completed in fn_epoch_listeners:
                fn_on_epoch_completed(epoch, self)
            if not self.scheduler is None: 
                self.scheduler.step()
            epoch += 1
        
    def train(self, batch):
        self.optimizer.zero_grad()
        token_loss = self.calculate_loss(batch)
        token_loss.backward()
        self.optimizer.step()    
        return token_loss.item() 

    def calculate_loss(self, batch):
        (image_features, caption_inputs, caption_targets, caption_lengths) = batch
        image_features = image_features.to(self.device) 
        caption_inputs = caption_inputs.to(self.device)
        caption_targets = caption_targets.to(self.device)
        caption_lengths = caption_lengths.to(self.device)

        output_probs, *k = self.decoder(
            image_features, caption_inputs, caption_lengths, device = self.device
        )    
        loss = self.loss_criterion(output_probs.permute(0, 2, 1), caption_targets)
        if len(k) > 1: # HACK
            # Add doubly stochastic attention regularization for show_attend_tell
            alphas = k[1]
            loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        return loss

    def calculate_validation_loss(self, val_data):
        # TODO: use teacher forcing or not?
        # TODO: take token length into account?
        self.decoder.eval()
        with torch.no_grad():
            total_loss = 0.
            total_tokens = 0
            for batch in val_data:
                token_loss = self.calculate_loss(batch)
                token_length = 1 #targets.size()[1] - 1
                total_loss += token_length * token_loss.item()
                total_tokens += token_length
        return total_loss / total_tokens

    # TODO: move to separate classes (SGD_builder, adam_builder)
    def set_optimizer(self):
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.decoder.parameters(), lr = self.learning_rate)
            self.scheduler = MultiStepLR(
                self.optimizer, 
                milestones = self.lr_decay, 
                gamma = 0.5)
        if self.optimizer_type == 'ADAM':
            self.optimizer = optim.Adam(self.decoder.parameters(), lr = self.learning_rate)
    
