import torch

from ncg.reporting.loss_collector import LossPlotter

def plot_losses(fpath_losses, fpath_plot_epoch_loss = '', fpath_plot_batch_loss = ''):
    loss_collector = torch.load(fpath_losses)
    loss_plotter = LossPlotter(loss_collector)
    if (fpath_plot_epoch_loss):
        loss_plotter.plotBatchLosses(fpath_plot_batch_loss)
    if (fpath_plot_batch_loss):
        loss_plotter.plotEpochLosses(fpath_plot_epoch_loss)
