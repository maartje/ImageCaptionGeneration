#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from ncg.preprocessor import preprocess_images

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    preprocess_opts(parser)

    opt = parser.parse_args()

    return opt

def preprocess_opts(parser):
    group = parser.add_argument_group('Preprocessor')
    group.add_argument(
        '-dataset', 
        help = "Name of the directory under 'data' containing the training and validation data",
        default = "flickr")
    group.add_argument(
        '-encoder_model', 
        help = "Name of the encoder model, should be in ['resnet18', 'resnet152', 'vgg16'] ",
        default = "resnet18")
    group.add_argument(
        '-encoder_layer', 
        help = "Name of the encoder layer used to extract features for the images",
        default = "avgpool")

def main():
    opt = parse_args()
    input_dir = f'data/{opt.dataset}/images'
    output_dir = f'output/{opt.dataset}/image_ecodings/{opt.encoder_model}_{opt.encoder_layer}'

    print(f"Encoding images as feature vectors ...")
    preprocess_images(input_dir, output_dir, opt.encoder_model, opt.encoder_layer)
    print(f"{len(os.listdir(output_dir))} images encoded ...")
    
    return output_dir

if __name__ == "__main__":
    main()

