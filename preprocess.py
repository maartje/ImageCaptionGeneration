#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob

import ncg.preprocess as pp

def preprocess_images(config):
    encoder_model = config['encoder_model']
    encoder_layer = config['encoder_layer']
    fpaths = glob.glob(config['fpattern_images'])
    output_dir_images = os.path.join(config['output_dir'], f"{encoder_model}_{encoder_layer}")
    print_info_every = int(config['print_info_every'])
#    print(fpaths)
#    print(fpaths_out)
#    print(list(fpaths))
#    print(list(fpaths_out))
#    print(encoder_model)
#    print(encoder_layer)
#    print(print_info_every)

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    pp.preprocess_images(fpaths, output_dir_images, 
                      encoder_model, encoder_layer, 
                      print_info_every)

def preprocess_descriptions(config):
    fpaths_train = glob.glob(config['fpattern_captions_train'])
    fpaths_val = glob.glob(config['fpattern_captions_val'])
    fpaths_train_out = [pt_fpath_out(config['output_dir'], fpath) for fpath in fpaths_train]
    fpaths_val_out = [pt_fpath_out(config['output_dir'], fpath) for fpath in fpaths_val]
    fpath_vocab_out = os.path.join(config['output_dir'], config['fname_vocab_out'])
    min_occurrences = int(opt.min_occurences)
    
#    print(fpaths_train)
#    print(fpaths_val)
#    print(fpaths_train_out)
#    print(fpaths_val_out)
#    print(fpath_vocab_out)
#    print(min_occurrences)

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
        
    pp.preprocess_text_files(fpaths_train, fpaths_val, 
                          fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                          min_occurrences)        

def pt_fpath_out(output_dir, fpath):
    fname = os.path.basename(fpath)
    return os.path.join(output_dir, f'{fname}.pt')

def main():
    description = 'Generate image encodings and indices vectors that represent image descriptions'
    config = get_configuration('preprocess', description = description)
    if not config['descriptions_only']:
        preprocess_images(opt)
    if not config['images_only']:
        preprocess_descriptions(config)

if __name__ == "__main__":
    main()

