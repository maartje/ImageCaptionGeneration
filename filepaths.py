import os
import glob

from ncg.io.file_helpers import read_lines

def get_file_paths(config):
    dataset = config["dataset"]
    output_dir = config["output_dir"]
    data_dir = os.path.join("data", dataset)
    captions_dir = os.path.join(data_dir, "captions", "en")
    images_dir = os.path.join(data_dir, "images")
    image_splits_dir = os.path.join(data_dir, "image_splits")
    output_dir_preprocess = os.path.join(output_dir, dataset, "preprocess")
    fname_vocab = config['fname_vocab']
    
    fpattern_captions_train = os.path.join(captions_dir, config['fpattern_captions_train'])
    fpattern_captions_val = os.path.join(captions_dir, config['fpattern_captions_val'])
    fpaths_captions_train = glob.glob(fpattern_captions_train)
    fpaths_captions_val = glob.glob(fpattern_captions_val)
    fpaths_caption_vectors_train = [
        pt_fpath_out(output_dir_preprocess, fpath) for fpath in fpaths_captions_train]
    fpaths_caption_vectors_val = [
        pt_fpath_out(output_dir_preprocess, fpath) for fpath in fpaths_captions_val]
    fpath_vocab = os.path.join(output_dir_preprocess, fname_vocab)

    
    return {
        'output' : output_dir,
        'captions_train' : fpaths_captions_train,
        'captions_val' : fpaths_captions_val,
        'caption_vectors_train' : fpaths_caption_vectors_train,
        'caption_vectors_val' : fpaths_caption_vectors_val,
        'vocab' : fpath_vocab
    }

def fpaths_image_split(dir_images, fpath_image_split, is_encoded = False): 
    fnames_image_split = read_lines(fpath_image_split)
    added_ext = ".pt" if is_encoded else ""
    return [os.path.join(dir_images, f'{fname}{added_ext}') for fname in fnames_image_split]

def pt_fpath_out(output_dir, fpath):
    fname = os.path.basename(fpath)
    return os.path.join(output_dir, f'{fname}.pt')

