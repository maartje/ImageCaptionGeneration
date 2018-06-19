import torch
from datetime import datetime
import os

from ncg.text_processing.textmapper import TextMapper
from ncg.image_processing.image_encoder import ImageEncoder
from ncg.debug_helpers import format_duration
import ncg.file_helpers as fh
from PIL import Image


def preprocess_images(input_dir, output_dir, 
                      encoder_model, encoder_layer, 
                      print_info_every = 1000):
    
    fnames = os.listdir(input_dir) # TODO: filepath because of train/val/test split or demo
    fpaths = [os.path.join(input_dir, fname) for fname in fnames]
    fpaths_out = [os.path.join(output_dir, f'{fname}.pt') for fname in fnames]

    # TODO check if outfiles do not exists unless --overwrite
    fh.ensure_path_exists(output_dir)

    image_encoder = ImageEncoder(encoder_model, encoder_layer)
    image_encoder.load_model()
    start_time = datetime.now()
    print(f'\n({format_duration(start_time, datetime.now())}) Start encoding images ...')
    for i, (fpath, fpath_out) in enumerate(zip(fpaths, fpaths_out)):
        img = Image.open(fpath)
        features = image_encoder.encode_image(img) # TODO: batch encoding?
        torch.save(features, fpath_out)
        if print_info_every and ((i+1) % print_info_every == 0):
            print (f'({format_duration(start_time, datetime.now())}) {i} images encoded')
    print(f'({format_duration(start_time, datetime.now())}) End encoding images ...')
    print(f"Created image embedding files stored in '{output_dir}'")

def preprocess_text_files(fpaths_train, fpaths_val, 
                          fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                          min_occurence = 2):
    fh.ensure_paths_exist(fpaths_train_out + fpaths_val_out + [fpath_vocab_out])
    # TODO: check if out files do not exist unless --overwrite
    
    start_time = datetime.now()
    mapper = build_and_save_vocabulary(fpaths_train, fpath_vocab_out, min_occurence, start_time)
    build_and_save_sentence_vectors(fpaths_train + fpaths_val, 
                                    fpaths_train_out + fpaths_val_out, 
                                    mapper, start_time)

def build_and_save_vocabulary(fpaths_train, fpath_vocab_out, min_occurence, start_time):
    print(f'\n({format_duration(start_time, datetime.now())}) Start building vocabulary ...')
    sentences_train = fh.read_lines_multiple_files(fpaths_train)         
    mapper = TextMapper()
    mapper.build(sentences_train, min_occurence)
    torch.save(mapper, fpath_vocab_out)
    duration_str = format_duration(start_time, datetime.now())
    print(f'({duration_str})    Saved vocabulary file at {fpath_vocab_out}')
    print(f'({format_duration(start_time, datetime.now())}) Finished building vocabulary ...')
    return mapper

def build_and_save_sentence_vectors(fpaths, fpaths_out, mapper, start_time):
    duration_str = format_duration(start_time, datetime.now())
    print(f'\n({duration_str}) Start building index vectors from descriptions ...')
    for (fpath, fpath_out) in zip(fpaths, fpaths_out):
        sentence_vectors = [mapper.sentence2indices(sentence) for sentence in fh.read_lines(fpath)]
        torch.save(sentence_vectors, fpath_out)
        duration_str = format_duration(start_time, datetime.now())
        print(f'({duration_str})    Saved indices file at {fpath_out}')
    duration_str = format_duration(start_time, datetime.now())
    print(f'({duration_str}) Finished building index vectors from descriptions ...')
    

    

