import torch
from os import system

import ncg.reporting.plots as p
from ncg.reporting.bleu import calculate_bleu

def plot_losses(fpath_losses, fpath_plot_epoch_loss = '', fpath_plot_batch_loss = ''):
    (
        batch_intervals, 
        batch_losses, 
        intervals_train, 
        losses_train, 
        intervals_val, 
        losses_val
    ) = torch.load(fpath_losses)
    p.plotBatchLosses(batch_intervals, batch_losses, fpath_plot_batch_loss)
    p.plotEpochLosses(
        intervals_train, losses_train, intervals_val, losses_val, fpath_plot_epoch_loss)

def plot_bleu_scores(fpath_bleu_scores, fpath_plot_bleu_scores):
    (intervals, bleu_scores) = torch.load(fpath_bleu_scores)
    p.plotBleuScores(intervals, bleu_scores, fpath_plot_bleu_scores)
        
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

