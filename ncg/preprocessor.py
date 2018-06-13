import os
from PIL import Image
import torch

from ncg.image_encoder import get_image_encoder, encode_image

def preprocess_images(input_dir, output_dir, encoder_model, encoder_layer):
    model, layer = get_image_encoder(encoder_model, encoder_layer)
    os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        img = Image.open(fpath)
        features = encode_image(model, layer, img)
        fpath_out = os.path.join(output_dir, f'{fname[:-4]}.pt')
        torch.save(features, fpath_out)

