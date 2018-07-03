#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import itertools
import glob

import ncg.preprocess as pp

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    preprocess_opts(parser)
    opt = parser.parse_args()
    return opt

def preprocess_opts(parser):
    group = parser.add_argument_group('Preprocessor')

    # general
    group.add_argument(
        '--output_dir', 
        help = "Path to output dir for the index vectors representing image descriptions ",
        default = os.path.join("output", "demo"))
    group.add_argument(
        '--descriptions_only', 
        help = "Preprocess the image descriptions and not the images",
        action='store_true')
    group.add_argument(
        '--images_only', 
        help = "Preprocess the images and not the image descriptions",
        action='store_true')
        #TODO: --vocab, --overwrite

    # image processing
    # TODO: image_selection: Pathname pattern to files containing the filenames of the images to be processed 
    group.add_argument(
        '--fpattern_images', 
        help = "Pathname pattern to the image files ",
        default = os.path.join("data", "demo", "images", "*.jpg"))
    group.add_argument(
        '--encoder_model', 
        help = "Name of the encoder model, should be in ['resnet18', 'resnet152', 'vgg16'] ",
        default = "resnet18")
    group.add_argument(
        '--encoder_layer', 
        help = "Name of the encoder layer used to extract features for the images",
        default = "avgpool")
    group.add_argument(
        '--print_info_every', 
        help = "Print info when given number of images are processed ",
        default = 1000)

    # text processing
    group.add_argument(
        '--fpattern_captions_train', 
        help = "Pathname pattern to the files containing descriptions used for training",
        default = os.path.join("data", "demo", "captions", "en", "train.[0-9].en"))
    group.add_argument(
        '--fpattern_captions_val', 
        help = "Pathname pattern to the files containing descriptions used for validation",
        default = os.path.join("data", "demo", "captions", "en", "val.[0-9].en"))
    # TODO: filepath_vocab?
    group.add_argument(
        '--fname_vocab_out', 
        help = "Filename for vocabulary output file",
        default = "vocab.pt")
    group.add_argument(
        '--min_occurences', 
        help = "Minimal occurrence in training set to be included in vocabulary",
        default = 2)

def preprocess_images(opt):
    encoder_model = opt.encoder_model
    encoder_layer = opt.encoder_layer
    fpaths = glob.glob(opt.fpattern_images)
    output_dir_images = os.path.join(opt.output_dir, f"{encoder_model}_{encoder_layer}")
    print_info_every = int(opt.print_info_every)
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

def preprocess_descriptions(opt):
    fpaths_train = glob.glob(opt.fpattern_captions_train)
    fpaths_val = glob.glob(opt.fpattern_captions_val)
    fpaths_train_out = [pt_fpath_out(opt.output_dir, fpath) for fpath in fpaths_train]
    fpaths_val_out = [pt_fpath_out(opt.output_dir, fpath) for fpath in fpaths_val]
    fpath_vocab_out = os.path.join(opt.output_dir, opt.fname_vocab_out)
    min_occurrences = int(opt.min_occurences)
    
#    print(fpaths_train)
#    print(fpaths_val)
#    print(fpaths_train_out)
#    print(fpaths_val_out)
#    print(fpath_vocab_out)
#    print(min_occurrences)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        
    pp.preprocess_text_files(fpaths_train, fpaths_val, 
                          fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                          min_occurrences)        

def main():
    opt = parse_args()
    if not opt.descriptions_only:
        preprocess_images(opt)
    if not opt.images_only:
        preprocess_descriptions(opt)


if __name__ == "__main__":
    main()

