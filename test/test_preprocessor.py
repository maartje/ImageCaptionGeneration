"""
Tests for preprocessing text and image files
"""

import unittest
import mock # mock file access
import ncg.preprocessor as pp
import ncg.file_helpers

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

