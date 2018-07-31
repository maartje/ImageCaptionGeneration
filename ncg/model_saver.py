import torch
import os

class ModelSaver():

    def __init__(self, model, bleu_collector, fpath_model, fpath_model_best):
        self.model = model
        self.bleu_collector = bleu_collector
        self.fpath_model = fpath_model
        self.fpath_model_best = fpath_model_best
        
    def on_epoch_completed(self, epoch, trainer):
        torch.save(self.model, self.fpath_model % epoch)
        fname_prev_model = self.fpath_model % (epoch - 1)
        if os.path.exists(fname_prev_model):
            os.remove(fname_prev_model)
        if self.bleu_collector.bleu_val:
            max_bleu = max(self.bleu_collector.bleu_val)
            last_bleu = self.bleu_collector.bleu_val[-1]
            if last_bleu == max_bleu:
                torch.save(self.model, self.fpath_model_best)
                print (f"Best model so far saved on epoch '{epoch}' with BLEU score '{last_bleu}'")
                


