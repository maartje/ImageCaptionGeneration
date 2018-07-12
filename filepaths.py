import os
import glob

from ncg.io.file_helpers import read_lines

def get_file_paths(config):
    dataset = config["dataset"]
    output_dir = config["output_dir"]
    data_dir = os.path.join("data", dataset)
    captions_dir = os.path.join(data_dir, "captions", "en")
    output_dir_preprocess = os.path.join(output_dir, dataset, "preprocess")
    fname_vocab = config['fname_vocab']
    
    # preprocess
    fpattern_captions_train = os.path.join(captions_dir, config['fpattern_captions_train'])
    fpattern_captions_val = os.path.join(captions_dir, config['fpattern_captions_val'])
    fpaths_captions_train = glob.glob(fpattern_captions_train)
    fpaths_captions_val = glob.glob(fpattern_captions_val)
    fpaths_caption_vectors_train = [
        pt_fpath_out(output_dir_preprocess, fpath) for fpath in fpaths_captions_train]
    fpaths_caption_vectors_val = [
        pt_fpath_out(output_dir_preprocess, fpath) for fpath in fpaths_captions_val]
    fpath_vocab = os.path.join(output_dir_preprocess, fname_vocab)

    # train
    #### TODO use matrix files
    fpath_image_split_train = os.path.join(data_dir, "image_splits", "train_images.txt")
    fpath_image_split_val = os.path.join(data_dir, "image_splits", "val_images.txt")
    fpath_image_split_test = os.path.join(data_dir, "image_splits", "val_images.txt")
    dir_image_features = os.path.join(output_dir_preprocess, 'resnet18_avgpool')
    fpaths_image_features_train = fpaths_image_split(dir_image_features, fpath_image_split_train, True)
    fpaths_image_features_val = fpaths_image_split(dir_image_features, fpath_image_split_val, True)
    fpaths_image_features_test = fpaths_image_split(dir_image_features, fpath_image_split_test, True)
    
    output_dir_train = os.path.join(output_dir, dataset, "train")
    fpath_losses = os.path.join(output_dir_train, config['fname_losses'])
    fpath_model = os.path.join(output_dir_train, f'{config["model"]}.pt')

    # predict
    output_dir_predict = os.path.join(output_dir, dataset, "predict")
    fpath_predictions_val = os.path.join(output_dir_predict, 'predictions_val.txt')
    fpath_predictions_test = os.path.join(output_dir_predict, 'predictions_test.txt')
    
    return {
        'captions_train' : fpaths_captions_train,
        'captions_val' : fpaths_captions_val,
        'caption_vectors_train' : fpaths_caption_vectors_train,
        'caption_vectors_val' : fpaths_caption_vectors_val,
        'vocab' : fpath_vocab,
        'image_features_train' : fpaths_image_features_train,
        'image_features_val': fpaths_image_features_val,
        'image_features_test': fpaths_image_features_test,
        'losses' : fpath_losses,
        'model' : fpath_model,
        'predictions_test' : fpath_predictions_test,
        'predictions_val' : fpath_predictions_val
    }

def fpaths_image_split(dir_images, fpath_image_split, is_encoded = False): 
    fnames_image_split = read_lines(fpath_image_split)
    added_ext = ".pt" if is_encoded else ""
    return [os.path.join(dir_images, f'{fname}{added_ext}') for fname in fnames_image_split]

def pt_fpath_out(output_dir, fpath):
    fname = os.path.basename(fpath)
    return os.path.join(output_dir, f'{fname}.pt')

