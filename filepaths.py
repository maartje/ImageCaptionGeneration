import os

from ncg.io.file_helpers import read_lines

def fpaths_image_split(dir_images, fpath_image_split, is_encoded = False): 
    fnames_image_split = read_lines(fpath_image_split)
    added_ext = ".pt" if is_encoded else ""
    return [os.path.join(dir_images, f'{fname}{added_ext}') for fname in fnames_image_split]

