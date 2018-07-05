"""
Tests for loading the data
"""

import unittest
import mock
from torch.utils import data
import torch

from ncg.io.embedding_description_dataset import EmbeddingDescriptionDataset
from ncg.io.embedding_description_dataset import EmbeddingDescriptionGroupedDataset


image_files = ['im1.pt', 'im2.pt', 'im3.pt']
caption_files = ['c1.pt', 'c2.pt']

def mock_torch_load(fpath):
    if fpath in image_files:
        return torch.FloatTensor([image_files.index(fpath)] * 5)
    if fpath in caption_files:
        i = caption_files.index(fpath)
        return [[0,0, i], [1,1,1,1,1, i], [2,2,2,i]]

class TestEmbeddingDescriptionDataloader(unittest.TestCase):

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def setUp(self, torch_load):
        self.ic_dataset = EmbeddingDescriptionDataset(
            image_files,
            caption_files
        )

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def test_data_set(self, torch_load):
        self.assertEqual(len(image_files)*len(caption_files), len(self.ic_dataset))

        expected_embeddings = [ 
            mock_torch_load(image_files[0]),
            mock_torch_load(image_files[1]),
            mock_torch_load(image_files[2]),
            mock_torch_load(image_files[0]),
            mock_torch_load(image_files[1]),
            mock_torch_load(image_files[2]),
        ]
        expected_descriptions = [
            mock_torch_load(caption_files[0])[0],
            mock_torch_load(caption_files[0])[1],
            mock_torch_load(caption_files[0])[2],
            mock_torch_load(caption_files[1])[0],
            mock_torch_load(caption_files[1])[1],
            mock_torch_load(caption_files[1])[2]
        ]
        for i in range(len(self.ic_dataset)):
            e, d = self.ic_dataset[i]
            self.assertEqual(e[0], expected_embeddings[i][0])
            self.assertEqual(d[0], expected_descriptions[i][0])
            self.assertEqual(d[-1], expected_descriptions[i][-1])
            
        self.assertEqual(torch_load.call_count, 6) # 3 images, 2 captions per image


class TestEmbeddingDescriptionGroupedDataset(unittest.TestCase):

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def setUp(self, torch_load):
        self.ic_dataset = EmbeddingDescriptionGroupedDataset(
            image_files,
            caption_files
        )        

    @mock.patch('torch.load', side_effect = mock_torch_load)
    def test_data_set(self, torch_load):
        self.assertEqual(len(image_files)*len(caption_files), len(self.ic_dataset))

        expected_embeddings = [ 
            mock_torch_load(image_files[0]),
            mock_torch_load(image_files[0]),
            mock_torch_load(image_files[1]),
            mock_torch_load(image_files[1]),
            mock_torch_load(image_files[2]),
            mock_torch_load(image_files[2]),
        ]
        expected_descriptions = [
            mock_torch_load(caption_files[0])[0],
            mock_torch_load(caption_files[1])[0],
            mock_torch_load(caption_files[0])[1],
            mock_torch_load(caption_files[1])[1],
            mock_torch_load(caption_files[0])[2],
            mock_torch_load(caption_files[1])[2]
        ]
        for i in range(len(self.ic_dataset)):
            e, d = self.ic_dataset[i]
            self.assertEqual(e[0], expected_embeddings[i][0])
            self.assertEqual(d[0], expected_descriptions[i][0])
            self.assertEqual(d[-1], expected_descriptions[i][-1])

        self.assertEqual(torch_load.call_count, 3) # 3 images, cached for multiple captions



if __name__ == '__main__':
    unittest.main()


