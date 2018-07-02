"""
Tests for the loss collector that stores epoch losses and batch losses
"""

import unittest
import mock

from ncg.reporting.loss_collector import LossCollector, LossPlotter

class TestLossCollector(unittest.TestCase):

    def setUp(self):
        self.epoch_losses = [
            [1,2,3,4,5,6,7,8,9],
            [10,20,30,40,50, 60, 70, 80, 90]]
        self.epoch_losses_val = [22, 33]
        self.initial_val_loss = 11
        
    def test_process_loss_with_partial(self):
        loss_collector = LossCollector(4)
        self.process_losses(loss_collector)        
        epoch_losses_expected = [5, 50]
        batch_losses_expected = [[2.5, 6.5, 9], [25, 65, 90]]        
        self.assertEqual(epoch_losses_expected, loss_collector.epoch_losses_train)
        self.assertEqual(batch_losses_expected, loss_collector.batch_losses_train)
        self.assertEqual(len(self.epoch_losses[0]), loss_collector.epoch_size)
        self.assertEqual(1, loss_collector.batch_loss_size_last)


    def test_process_loss_without_partial(self):
        loss_collector = LossCollector(3)
        self.process_losses(loss_collector)
        epoch_losses_expected = [5, 50]
        batch_losses_expected = [[2,5,8], [20,50,80]]
        self.assertEqual(batch_losses_expected, loss_collector.batch_losses_train)
        self.assertEqual(3, loss_collector.batch_loss_size_last)

    
    @mock.patch('matplotlib.pyplot.savefig')
    def test_plot_losses(self, save_fig):
        loss_collector = LossCollector(4)
        self.process_losses(loss_collector)

        self.assertEqual(
            ([0, 9, 18], [11, 22, 33]), 
            loss_collector.plot_values_epoch_losses_val())
        self.assertEqual(
            ([9, 18], [5.0, 50.0]), 
            loss_collector.plot_values_epoch_losses_train())
        self.assertEqual(
            ([4, 8, 9, 13, 17, 18], [2.5, 6.5, 9.0, 25.0, 65.0, 90.0]),
            loss_collector.plot_values_batch_losses_train())

        loss_collector_no_partial = LossCollector(3)
        self.process_losses(loss_collector_no_partial)
        self.assertEqual(
            ([3, 6, 9, 12, 15, 18], [2, 5, 8, 20, 50, 80]),
            loss_collector_no_partial.plot_values_batch_losses_train())


        plotter = LossPlotter(loss_collector)
        plotter.plotEpochLosses('mj_epoch.png')
        plotter.plotBatchLosses('mj_batch.png')
        
    
    def process_losses(self, loss_collector):
        loss_collector.initial_validation_loss = self.initial_val_loss
        for e, losses_in_epoch in enumerate(self.epoch_losses):
            for i, loss in enumerate(losses_in_epoch):
                loss_collector.on_batch_completed(e, i, loss)
            loss_collector.on_epoch_completed(e, i, self.epoch_losses_val[e])
            
     


if __name__ == '__main__':
    unittest.main()


