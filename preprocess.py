#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import itertools
import glob

from ncg.preprocessor import preprocess_text_files, preprocess_images

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
        '--filepaths_images', 
        help = "Pathname pattern to the image files ",
        default = os.path.join("data", "demo", "images", "*.png"))
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
        '--filepaths_train', 
        help = "Pathname pattern to the files containing descriptions used for training",
        default = os.path.join("data", "demo", "train.[0-9].en"))
    group.add_argument(
        '--filepaths_val', 
        help = "Pathname pattern to the files containing descriptions used for validation",
        default = os.path.join("data", "demo", "val.[0-9].en"))
    # TODO: filepath_vocab?
    group.add_argument(
        '--fname_vocab_out', 
        help = "Filename for vocabulary output file",
        default = "vocab.pt")
    group.add_argument(
        '--min_occurrences', 
        help = "Minimal occurrence in training set to be included in vocabulary",
        default = 2)

def pt_fpath_out(output_dir, fpath):
    fname = os.path.basename(fpath)
    return os.path.join(output_dir, f'{fname}.pt')

def preprocess_images(opt):
    encoder_model = opt.encoder_model
    encoder_layer = opt.encoder_layer
    fpaths = glob.iglob(opt.filepaths_images)
    output_dir_images = os.path.join(opt.output_dir, f"{encoder_model}_{encoder_layer}")
    fpaths_out = (
        pt_fpath_out(output_dir_images, fpath) for fpath in glob.iglob(opt.filepaths_images))
    print_info_every = opt.print_info_every
    print(fpaths)
    print(fpaths_out)
    print(list(fpaths))
    print(list(fpaths_out))
    print(encoder_model)
    print(encoder_layer)
    print(print_info_every)


def preprocess_descriptions(opt):
    filepaths_train = glob.glob(opt.filepaths_train)
    filepaths_val = glob.glob(opt.filepaths_val)
    fpaths_train_out = [pt_fpath_out(opt.output_dir, fpath) for fpath in filepaths_train]
    fpaths_val_out = [pt_fpath_out(opt.output_dir, fpath) for fpath in filepaths_val]
    fpath_vocab_out = os.path.join(opt.output_dir, opt.fname_vocab_out)
    min_occurrences = opt.min_occurrences
    
    print(filepaths_train)
    print(filepaths_val)
    print(fpaths_train_out)
    print(fpaths_val_out)
    print(fpath_vocab_out)
    print(min_occurrences)


def main():
    opt = parse_args()
    print(opt)
    print()
    if not opt.descriptions_only:
        preprocess_images(opt)
    if not opt.images_only:
        preprocess_descriptions(opt)


if __name__ == "__main__":
    main()

