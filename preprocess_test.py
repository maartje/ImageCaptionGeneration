from datetime import datetime
from ncg.preprocess import build_and_save_sentence_vectors
import torch
import os

if __name__ == "__main__":
    dir_in = os.path.join('data', 'flickr30k', 'captions', 'en')
    fnames = [
        'test_2016.1.en', 'test_2016.2.en', 'test_2016.3.en', 'test_2016.4.en', 'test_2016.5.en'
    ]
    fpaths_in = [ os.path.join(dir_in, fname) for fname in fnames] 
    dir_out = os.path.join('output', 'flickr30k', 'preprocess')
    fpaths_out = [ os.path.join(dir_out, f'{fname}.pt') for fname in fnames]
    fpath_vocab = os.path.join(dir_out, 'vocab.pt')
    mapper = torch.load(fpath_vocab)
    build_and_save_sentence_vectors(fpaths_in, fpaths_out, mapper, datetime.now())
