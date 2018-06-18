#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import itertools

from ncg.preprocessor preprocess_text_files
#import build_and_save_image_features

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
        '--dataset', 
        help = "Name of the directory under 'data' containing the training and validation data",
        default = "demo")
    group.add_argument(
        '--images_only', 
        help = "Preprocess the images and not the image descriptions",
        action='store_true')
    group.add_argument(
        '--descriptions_only', 
        help = "Preprocess the image descriptions and not the images",
        action='store_true')

    # image processing
    group.add_argument(
        '--encoder_model', 
        help = "Name of the encoder model, should be in ['resnet18', 'resnet152', 'vgg16'] ",
        default = "resnet18")
    group.add_argument(
        '--encoder_layer', 
        help = "Name of the encoder layer used to extract features for the images",
        default = "avgpool")

    # text processing
    group.add_argument(
        '--regex_train', 
        help = "Regular expression to match caption files used for training",
        default = 'train\.[0-9]+\.en')
    group.add_argument(
        '--regex_val', 
        help = "Regular expression to match caption files used for validation",
        default = 'val\.[0-9]+\.en')
    group.add_argument(
        '--min_occurrences', 
        help = "Minimal occurrence in training set to be included in vocabulary",
        default = 2)

def preprocess_images(opt):
    input_dir = f'data/{opt.dataset}/images'
    output_dir = f'output/{opt.dataset}/image_ecodings/{opt.encoder_model}_{opt.encoder_layer}'

    print(f"Encoding images as feature vectors ...")
    build_and_save_image_features(input_dir, output_dir, opt.encoder_model, opt.encoder_layer, 
                                  print_info_every = 1000)
    print(f"{len(os.listdir(output_dir))} images encoded and saved in {output_dir}")

    return output_dir

def preprocess_image_descriptions(opt):
    data_dir = f"data/{opt.dataset}"
    output_dir = f"output/{opt.dataset}"
    fpath_vocab = f"output/{opt.dataset}/vocab.pt"

    fnames_train = filenames(data_dir, opt.regex_train)
    fnames_val = filenames(data_dir, opt.regex_val)
    fpaths_train = [os.path.join(data_dir, fname) for fname in fnames_train]
    fpaths_val = [os.path.join(data_dir, fname) for fname in fnames_val]
    fpaths_train_out = [os.path.join(output_dir, f'{fname}.pt') for fname in fnames_train]
    fpaths_val_out = [os.path.join(output_dir, f'{fname}.pt') for fname in fnames_val]

    print(fpaths_train)
    print(fpaths_val)
    build_and_save_sentence_vectors_and_vocabulary(fpaths_train, fpaths_val, 
                                                   fpaths_train_out, fpaths_val_out, 
                                                   fpath_vocab, opt.min_occurrences)


def filenames(dir_path, regex):
    pattern = re.compile(regex)
    lstdir = os.listdir(dir_path)
    for fname in lstdir: 
        if pattern.match(fname):
            yield fname

def filepaths(dir_path, fnames, new_extension = None):
    for fname in fnames:
        fname_new = f'{fname}.{new_extension}' if new_extension else fname
        yield os.path.join(output_dir, fname_new)

def main():
    opt = parse_args()
    print(opt)
    if not opt.descriptions_only:
        preprocess_images(opt)
    if not opt.images_only:
        preprocess_image_descriptions(opt)


if __name__ == "__main__":
    main()

