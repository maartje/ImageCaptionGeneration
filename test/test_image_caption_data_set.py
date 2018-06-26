"""
Tests for loading the data
"""

import unittest
import mock
from torch.utils import data
import torch

from ncg.image_caption_dataset import ImageCaptionDataset

image_files = ['im1.pt', 'im2.pt', 'im3.pt']
caption_files = ['c1.pt', 'c2.pt', 'c3.pt', 'c4.pt']

def mock_torch_load(fpath):
    if fpath in image_files:
        return torch.FloatTensor([image_files.index(fpath)] * 5)
    if fpath in caption_files:
        i = caption_files.index(fpath)
        return [[0,0, i], [1,1,1,1,1, i], [2,2,2,i]]

class TestImageCaptionDataloader(unittest.TestCase):

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def setUp(self, torch_load):
        self.ic_dataset = ImageCaptionDataset(
            image_files,
            caption_files
        )

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def test_data_set(self, torch_load):

        # dataset contains all image/caption combinations    
        self.assertEqual(len(image_files)*len(caption_files), len(self.ic_dataset))
        img_expected = mock_torch_load(image_files[2])
        caption_expected = mock_torch_load(caption_files[1])[2]
        (img_actual, caption_actual) = self.ic_dataset[5]
        self.assertEqual(img_actual[0], img_expected[0]) 
        self.assertEqual(caption_actual[0], caption_expected[0]) 
        self.assertEqual(caption_actual[-1], caption_expected[-1]) 

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def test_data_loading(self, torch_load):
        dataLoader = data.DataLoader(self.ic_dataset)

        # generates all (image, caption) combinations 
        self.assertEqual(len(list(dataLoader)), len(image_files)*len(caption_files))
        self.assertEqual(image_files * len(caption_files), 
                         [fp for (fp,), _ in torch_load.call_args_list])
#        for i, c in dataLoader:
#            print()
#            print ('i', i)
#            print ('c', c)

#    @mock.patch('torch.load', side_effect = mock_torch_load)
#    def test_data_loading(self, torch_load):
#        dataLoader = data.DataLoader(self.ic_dataset, batch_size = 2)

        # generates all (image, caption) combinations 
        #self.assertEqual(len(list(dataLoader)), len(image_files)*len(caption_files))
        #self.assertEqual(image_files * len(caption_files), 
        #                 [fp for (fp,), _ in torch_load.call_args_list])
#        for i, c in dataLoader:
#            print()
#            print ('i', i)
#            print ('c', c)

if __name__ == '__main__':
    unittest.main()


