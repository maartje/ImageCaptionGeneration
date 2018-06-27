"""
Tests for the image encoder that maps images to ebedding vectors
"""

import unittest
from ncg.data_processing.image_encoder import ImageEncoder
import numpy
from PIL import Image

class TestImageEncoder(unittest.TestCase):

    def setUp(self):
        self.image_encoder = ImageEncoder('resnet18', 'avgpool')
        self.image_encoder.load_model()
        
    def test_encode_image_large(self):
        # create random imagewidth, height
        img = self.create_random_image(620, 580)
        img_encoding = self.image_encoder.encode_image(img)
        
        self.assertEqual(list(img_encoding.size()), [512])
        self.assertTrue(sum(img_encoding.data) > 0)

    def test_encode_image_small(self):
        # create random imagewidth, height
        img = self.create_random_image(120, 134)
        img_encoding = self.image_encoder.encode_image(img)
        
        self.assertEqual(list(img_encoding.size()), [512])
        self.assertTrue(sum(img_encoding.data) > 0)

    def test_encode_images(self):
        img1 = self.create_random_image(620, 580)
        img2 = self.create_random_image(120, 134)
        images = [img1, img2]
        emb1 = self.image_encoder.encode_image(img1)
        emb2 = self.image_encoder.encode_image(img2)
        
        emb_all = self.image_encoder.encode_images(images)
        
        emb_all_1 = emb_all[0,:]
        emb_all_2 = emb_all[1,:]
        self.assertEqual(list(emb_all.size()), [2, 512])
        self.assertEqual(emb1[120], emb_all_1[120])
        self.assertEqual(emb2[110], emb_all_2[110])
                
    def create_random_image(self, width, height):
        imarray = numpy.random.rand(width, height, 3) * 255
        img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        return img

if __name__ == '__main__':
    unittest.main()



