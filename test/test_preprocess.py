"""
Tests for preprocessing text and image files
"""

import unittest
import mock # mock file access
import PIL.Image

import ncg.preprocess as pp
import ncg.data_processing.image_encoder as imenc
import ncg.io.file_helpers as fh
from test.test_helpers import generate_random_image

def mock_read_lines(fpath):
    sentences_per_file = {
        "train.1.en" : ['Hello world!', 'Hello world!', 'trainone'], 
        "train.2.en" : ['Foo, bar.', 'traintwo'],
        "val.1.en" : ['Hello Foo', 'Hello X'], 
        "val.2.en" : ['Hello foo. Hello bar.']
    }
    return (s for s in sentences_per_file[fpath])
    
    
class TestPreprocessor(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('torch.save')
    @mock.patch('PIL.Image.open')
    def test_preprocess_images(self, open_image, torch_save, pprint = None):
        open_image.return_value = generate_random_image() 
                    
        fpaths = ['im1.png', 'im2.png']
        out_dir = "out" 
        encoder_model = 'resnet18'  
        encoder_layer = 'avgpool' 
        
        pp.preprocess_images(fpaths, out_dir, 
                          encoder_model, encoder_layer, print_info_every = 1)
                                          
        # check encoding size
        encodings = [embedding for (embedding, fp), _ in torch_save.call_args_list]
        self.assertTrue(all([e.size(0) == 512 for e in encodings]))

        # check fpaths embeddings saved
        fpaths_out = ['out/im1.png.pt', 'out/im2.png.pt']  
        fpaths_saved = [fp for (embedding, fp), _ in torch_save.call_args_list]
        self.assertEqual(fpaths_out, fpaths_saved)
        
                      
    @mock.patch('builtins.print')
    @mock.patch('torch.save')
    @mock.patch('ncg.io.file_helpers.read_lines', side_effect=mock_read_lines)
    def test_preprocess_text_files(self, read_lines, torch_save, pprint=None):
        fpaths_train = ["train.1.en", "train.2.en"] 
        fpaths_val = ["val.1.en", "val.2.en"]
        fpaths_train_out = ["train.1.en.pt", "train.2.en.pt"]
        fpaths_val_out = ["val.1.en.pt", "val.2.en.pt"]
        fpath_vocab_out = "vocab.pt"
        min_occurence = 1
        
        pp.preprocess_text_files(fpaths_train, fpaths_val, 
                                 fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                                 min_occurence)
       
        # saves text mapper at 'fpath_vocab_out'
        (tm, fp), _ =torch_save.call_args_list[0]
        self.assertEqual('TextMapper', type(tm).__name__)
        self.assertEqual(fpath_vocab_out, fp)

        # saves vector representations for all train and validation files
        # at fpath_train_out resp. fpath_val_out
        fpaths_save = [fp for (vectors, fp), _ in torch_save.call_args_list[1:]]
        self.assertEqual(fpaths_train_out + fpaths_val_out, fpaths_save) # check fpath out

        vectors_save = [vectors for (vectors, fp), _ in torch_save.call_args_list[1:]]
        self.assertEqual([
            tm.sentence2indices(s) for s in mock_read_lines(fpaths_val[0])
        ], vectors_save[len(fpaths_train)]) 
        self.assertEqual([
            tm.indices2sentence(v) for v in vectors_save[len(fpaths_train)]
        ], ['hello foo', 'hello']) # check vector representation

