#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import ensure_paths_exist
import ncg.preprocess as pp
from parse_config import get_configuration

#def preprocess_images(config):
#    encoder_model = config['encoder_model']
#    encoder_layer = config['encoder_layer']
#    fpaths = glob.glob(config['fpattern_images'])
#    output_dir_images = os.path.join(config['output_dir'], f"{encoder_model}_{encoder_layer}")
#    print_info_every = int(config['print_info_every'])
#    print(fpaths)
#    print(fpaths_out)
#    print(list(fpaths))
#    print(list(fpaths_out))
#    print(encoder_model)
#    print(encoder_layer)
#    print(print_info_every)

#    if not os.path.exists(output_dir_images):
#        os.makedirs(output_dir_images)

#    pp.preprocess_images(fpaths, output_dir_images, 
#                      encoder_model, encoder_layer, 
#                      print_info_every)

def preprocess_descriptions(config, filepaths):

    ensure_paths_exist[filepaths['vocab']]
        
    pp.preprocess_text_files(
        filepaths['captions_train'], filepaths['captions_val'], #in
        filepaths['caption_vectors_train'], filepaths['caption_vectors_val'], filepaths['vocab'], #out
        config['min_occurences']) #config


def main():
    description = 'Generate indices vectors that represent image descriptions'
    config, filepaths = get_configuration('preprocess', description = description)
    preprocess_descriptions(config, filepaths)
#    if not config['descriptions_only']:
#        preprocess_images(opt)
#    if not config['images_only']:
#        preprocess_descriptions(config)

if __name__ == "__main__":
    main()

