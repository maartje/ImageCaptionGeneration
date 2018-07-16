import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_image_features_descriptions(batch):
    PAD_index = 0 # TODO: Read from config
    
    transposed = list(zip(*batch))

    # pad, sort and stack captions
    captions = transposed[1]
    captions_input = [c[:-1] for c in captions] # remove EOS
    caption_lengths = np.array([len(c) for c in captions_input])
    sort_indices = np.argsort(caption_lengths)[::-1].copy()
    max_length = max(caption_lengths)
    
    captions_input_padded = [
        torch.cat(
            (c, torch.LongTensor([PAD_index] * (max_length - len(c))))
        )  for c in captions_input]      
    captions_input_collated = default_collate(captions_input_padded)
    captions_input_collated_sorted = captions_input_collated[sort_indices]

    captions_target = [c[1:] for c in captions] # remove SOS
    captions_target_padded = [
        torch.cat(
            (c, torch.LongTensor([PAD_index] * (max_length - len(c))))
        )  for c in captions_target]      
    captions_target_collated = default_collate(captions_target_padded)
    captions_target_collated_sorted = captions_target_collated[sort_indices]

    # sort and stack image encodings
    im_features = transposed[0]
    image_features_collated = default_collate(im_features)
    image_features_collated_sorted = image_features_collated[sort_indices]

    return [
        image_features_collated_sorted, 
        captions_input_collated_sorted, 
        captions_target_collated_sorted, 
        torch.LongTensor(caption_lengths[sort_indices])
    ]

