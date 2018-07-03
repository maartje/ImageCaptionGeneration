"""
Tests for the image encoder that maps images to ebedding vectors
"""

import unittest
import mock
from torch.utils import data
from test.test_helpers import generate_random_image

from ncg.data_processing.image_encoder import ImageEncoder
from ncg.io.image_dataset import ImageDataset

class TestImageEncoder(unittest.TestCase):

    def setUp(self):
        self.image_encoder = ImageEncoder('resnet18', 'avgpool')
        self.image_encoder.load_model()
        self.image_dataset = ImageDataset(['im1', 'im2', 'im3'])
    
    @mock.patch('PIL.Image.open')
    def test_encode_image_large(self, open_image):
        open_image.return_value = generate_random_image(640, 320) 
        
        # create random imagewidth, height
        dataloader = data.DataLoader(self.image_dataset)
        im, _ = next(iter(dataloader))
        img_encoding = self.image_encoder.encode(im)
        
        self.assertEqual(list(img_encoding.size()), [512])
        self.assertTrue(sum(img_encoding.data) > 0)

    @mock.patch('PIL.Image.open')
    def test_encode_image_small(self, open_image):
        open_image.return_value = generate_random_image(120, 134) 

        # create random imagewidth, height
        dataloader = data.DataLoader(self.image_dataset)
        im, _ = next(iter(dataloader))
        img_encoding = self.image_encoder.encode(im)
       
        self.assertEqual(list(img_encoding.size()), [512])
        self.assertTrue(sum(img_encoding.data) > 0)

    @mock.patch('PIL.Image.open')
    def test_encode_images(self, open_image):
        open_image.return_value = generate_random_image() 

        dataloader_1 = data.DataLoader(self.image_dataset, batch_size = 1)
        dataloader_2 = data.DataLoader(self.image_dataset, batch_size = 2)

        emb1 = self.image_encoder.encode(next(iter(dataloader_1))[0])
        emb2 = self.image_encoder.encode(next(iter(dataloader_1))[0])        
        emb_all = self.image_encoder.encode(next(iter(dataloader_2))[0])
        
        emb_all_1 = emb_all[0,:]
        emb_all_2 = emb_all[1,:]
        self.assertEqual(list(emb_all.size()), [2, 512])
        self.assertEqual(emb1[120], emb_all_1[120])
        self.assertEqual(emb2[110], emb_all_2[110])
                

if __name__ == '__main__':
    unittest.main()



