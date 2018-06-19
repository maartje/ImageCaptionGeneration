"""
Tests for preprocessing text and image files
"""

import unittest
import mock # mock file access
import ncg.preprocessor as pp
import ncg.file_helpers
import PIL.Image
import ncg.image_processing.image_encoder as imenc

def mock_read_lines(fpath):
    sentences_per_file = {
        "train.1.en" : ['Hello world!', 'Hello world!', 'trainone'], 
        "train.2.en" : ['Foo, bar.', 'traintwo'],
        "val.1.en" : ['Hello Foo', 'Hello X'], 
        "val.2.en" : ['Hello foo. Hello bar.']
    }
    return (s for s in sentences_per_file[fpath])

def mock_listdir(dir_path):
    return ['im1.png', 'im2.png']
    
def mock_encode_image(img):
    return f'Encode({img})'
    
def mock_open_image(path):
    return f'Open({path})'
    
class TestPreprocessor(unittest.TestCase):

    @mock.patch('builtins.print')
    @mock.patch('torch.save')
    @mock.patch('ncg.file_helpers.ensure_path_exists')
    @mock.patch('ncg.preprocessor.ImageEncoder')    
    @mock.patch('os.listdir', side_effect = mock_listdir)
    @mock.patch('PIL.Image.open', side_effect = mock_open_image)
    def test_preprocess_images(self, open_image, list_dir, image_encoder_class,
                               ensure_path, torch_save, pprint = None):
            
        image_encoder = image_encoder_class.return_value
        image_encoder.encode_image.side_effect = mock_encode_image
        
        input_dir = 'images' 
        output_dir = 'images_out' 
        encoder_model = 'resnet18'  
        encoder_layer = 'avgpool' 
        
        pp.preprocess_images(input_dir, output_dir, 
                          encoder_model, encoder_layer, print_info_every = 1)
                          
        # ensures that path exists for output dir
        fpaths_ensured = [fp for (fp,), _ in ensure_path.call_args_list]
        self.assertEqual([output_dir], fpaths_ensured)
        
        # verify that load model is called
        image_encoder.load_model.assert_called_with()

        # Create embeddings for all images by calling 'Image.open' and 'encode_image'
        encodings_saved = [embedding for (embedding, fp), _ in torch_save.call_args_list]
        encodings_expected = [ mock_encode_image(
            mock_open_image(f'{input_dir}/{fname}')) for fname in mock_listdir(input_dir)]
        self.assertEqual(encodings_expected, encodings_saved)

        # Save embeddings at fpaths out
        fpaths_saved = [fp for (embedding, fp), _ in torch_save.call_args_list]
        fpaths_expected = [f'{output_dir}/{fname}.pt' for fname in mock_listdir(input_dir)]
        self.assertEqual(fpaths_expected, fpaths_saved)
        
                      
    @mock.patch('builtins.print')
    @mock.patch('torch.save')
    @mock.patch('ncg.file_helpers.ensure_path_exists')
    @mock.patch('ncg.file_helpers.read_lines', side_effect=mock_read_lines)
    def test_preprocess_text_files(self, read_lines, ensure_path_exists, torch_save, pprint=None):
        fpaths_train = ["train.1.en", "train.2.en"] 
        fpaths_val = ["val.1.en", "val.2.en"]
        fpaths_train_out = ["train.1.en.pt", "train.2.en.pt"]
        fpaths_val_out = ["val.1.en.pt", "val.2.en.pt"]
        fpath_vocab_out = "vocab.pt"
        min_occurence = 1
        
        pp.preprocess_text_files(fpaths_train, fpaths_val, 
                                 fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                                 min_occurence)

        # ensures that path exists for all output files
        fpaths_ensured = [fp for (fp,), _ in ensure_path_exists.call_args_list]
        fpaths_out = fpaths_train_out + fpaths_val_out + [fpath_vocab_out]
        self.assertEqual(fpaths_out, fpaths_ensured)
        
        # saves text mapper at 'fpath_vocab_out'
        (tm, fp), _ =torch_save.call_args_list[0]
        self.assertEqual('TextMapper', type(tm).__name__)
        self.assertEqual(fpath_vocab_out, fp)

        # saves vector representations for all train and validation files
        # at fpath_train_out resp. fpath_val_out
        vectors_fpath_list = [(vectors, fp) for (vectors, fp), _ in torch_save.call_args_list[1:]]
        self.assertEqual(len(fpaths_train + fpaths_val), len(vectors_fpath_list)) # check length
        
        vectors_fpath_val1 = vectors_fpath_list[len(fpaths_train_out)]
        fpath_val1_save = vectors_fpath_val1[1]
        self.assertEqual(fpaths_val_out[0], fpath_val1_save) # check fpath out

        vectors_val1 = vectors_fpath_val1[0]
        self.assertEqual([
            tm.sentence2indices(s) for s in mock_read_lines(fpaths_val[0])
        ], vectors_val1) 
        self.assertEqual([
            tm.indices2sentence(v) for v in vectors_val1
        ], ['hello foo', 'hello UNKNOWN']) # check vector representation

