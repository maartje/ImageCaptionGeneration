import os
from PIL import Image
import torch

from ncg.image_encoder import get_image_encoder, encode_image
from ncg.text_processor import build_vocabulary, sentence2indices

def build_and_save_image_features(input_dir, output_dir, encoder_model, encoder_layer):
    model, layer = get_image_encoder(encoder_model, encoder_layer)
    os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        img = Image.open(fpath)
        features = encode_image(model, layer, img)
        fpath_out = os.path.join(output_dir, f'{fname}.pt')
        torch.save(features, fpath_out)

def build_and_save_sentence_vectors_and_vocabulary(fpaths_train, fpaths_val, 
                                                   fpaths_train_out, fpaths_val_out, 
                                                   fpath_vocab, min_occurrences):
    
    vocab = build_and_save_vocabulary(fpaths_train, fpath_vocab, min_occurrences)
    build_and_save_sentence_vectors(fpaths_train, fpaths_train_out, vocab)
    build_and_save_sentence_vectors(fpaths_val, fpaths_val_out, vocab)

def build_and_save_vocabulary(fpaths, fpath_vocab, min_occurrences):
    sentence_generator = _read_sentences(fpaths)
    vocab = build_vocabulary(sentence_generator, min_occurrences)
    _ensure_path_exists(fpath_vocab)
    torch.save(vocab, fpath_vocab)
    return vocab

def build_and_save_sentence_vectors(fpaths, fpaths_out, vocab):
    for fpath, fpath_out in zip(fpaths, fpaths_out):
        with open(fpath, 'r') as sentences:
            index_vectors = [sentence2indices(s, vocab) for s in sentences]
        _ensure_path_exists(fpath_out)
        torch.save(index_vectors, fpath_out)
        
def _read_sentences(fpaths):
    for fpath in fpaths:
        with open(fpath, 'r') as sentences:
            for sentence in sentences:
                yield sentence

def _ensure_path_exists(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
