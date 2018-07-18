import torch
from os import system

from ncg.reporting.loss_collector import LossPlotter, createLossCollectorFromDict
from ncg.reporting.bleu import calculate_bleu

def plot_losses(fpath_losses, fpath_plot_epoch_loss = '', fpath_plot_batch_loss = ''):
    loss_collector = createLossCollectorFromDict(torch.load(fpath_losses))
    loss_plotter = LossPlotter(loss_collector)
    if (fpath_plot_epoch_loss):
        loss_plotter.plotBatchLosses(fpath_plot_batch_loss)
    if (fpath_plot_batch_loss):
        loss_plotter.plotEpochLosses(fpath_plot_epoch_loss)
        
def calculate_metrics(fpaths_references, fpath_predicted, fpath_save_bleu):
    calculate_bleu(fpaths_references, fpath_predicted, fpath_save_bleu)
    
def compare_with_human_performance(fpaths_references, fpath_predicted, fpath_out):

    # human performance
    fpaths_references_str = ' '.join(fpaths_references[:-1])
    fpath_human = fpaths_references[-1]
    system(f'echo BLEU score human annotator based on {len(fpaths_references[:-1])} references >> {fpath_out}')
    system(
        f'./ncg/scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_human} >> {fpath_out}'
    )
    
    system(f'echo >> {fpath_out}')
    
    # model performance
    system(f'echo BLEU score model based on {len(fpaths_references[:-1])} references >> {fpath_out}')
    system(
        f'./ncg/scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_predicted} >> {fpath_out}'
    )

