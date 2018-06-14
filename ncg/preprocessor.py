import os
from PIL import Image
import torch

from ncg.image_encoder import get_image_encoder, encode_image
from ncg.text_processor import build_vocabulary, sentence2indices

def preprocess_images(input_dir, output_dir, encoder_model, encoder_layer):
    model, layer = get_image_encoder(encoder_model, encoder_layer)
    os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        img = Image.open(fpath)
        features = encode_image(model, layer, img)
        fpath_out = os.path.join(output_dir, f'{fname[:-4]}.pt')
        torch.save(features, fpath_out)

def preprocess_text(fpath_train, fpath_val, fpath_vocab_out, 
                    fpath_train_out, fpath_val_out, min_occurence = 2):
    vocab = build_and_save_vocab(fpath_train, fpath_vocab_out, min_occurence)
    build_and_save_index_vectors(fpath_train, fpath_train_out, vocab)
    build_and_save_index_vectors(fpath_val, fpath_val_out, vocab)
	
def build_and_save_vocab(fpath, fpath_out, min_occurence):
    with open(fpath_train, 'r') as sentences:
        vocab = build_vocabulary(sentences, min_occurence)
    torch.save(vocab, fpath_out)
    return vocab

def build_and_save_index_vectors(fpath, fpath_out, vocab):
    with open(fpath, 'r') as sentences:
        index_vectors = [sentence2indices(s, vocab, min_occurence) for s in sentences]
    torch.save(index_vectors, fpath_out)
    return index_vectors

