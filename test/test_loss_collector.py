"""
Tests for the loss collector that stores epoch losses and batch losses
"""

import unittest
from ncg.reporting.loss_collector import LossCollector
from datetime import datetime

class TestLossCollector(unittest.TestCase):

    def setUp(self):
        self.epoch_losses = [
            [1,2,3,4,5,6,7,8,9],
            [10,20,30,40,50, 60, 70, 80, 90]]

    def test_process_loss_with_partial(self):
        loss_collector = LossCollector(4, 1, datetime.now())
        self.process_losses(loss_collector)
        epoch_losses_expected = [5, 50]
        batch_losses_expected = [[2.5, 6.5, 9], [25, 65, 90]]
        self.assertEqual(epoch_losses_expected, loss_collector.epoch_losses_train)
        self.assertEqual(batch_losses_expected, loss_collector.batch_losses_train)
        self.assertEqual(len(self.epoch_losses[0]), loss_collector.epoch_size)
        self.assertEqual(1, loss_collector.batch_loss_size_last)

    def test_process_loss_without_partial(self):
        loss_collector = LossCollector(3, 1, datetime.now())
        self.process_losses(loss_collector)
        epoch_losses_expected = [5, 50]
        batch_losses_expected = [[2,5,8], [20,50,80]]
        self.assertEqual(epoch_losses_expected, loss_collector.epoch_losses_train)
        self.assertEqual(batch_losses_expected, loss_collector.batch_losses_train)
        self.assertEqual(len(self.epoch_losses[0]), loss_collector.epoch_size)
        self.assertEqual(0, loss_collector.batch_loss_size_last)

    def process_losses(self, loss_collector):
        for e, losses_in_epoch in enumerate(self.epoch_losses):
            for i, loss in enumerate(losses_in_epoch):
                loss_collector.on_batch_completed(e, i, loss)
            loss_collector.on_epoch_completed(e, i, -1)


if __name__ == '__main__':
    unittest.main()


