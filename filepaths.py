import os
import glob

from ncg.io.file_helpers import read_lines

def get_file_paths(config):
    config = replace_env_vars(config)
    main_dir = config["main_dir"]
    model_dir = config["model_dir"]

    # preprocess
    fpattern_captions_train = os.path.join(main_dir, 'data', config['fpattern_captions_train'])
    fpattern_captions_val = os.path.join(main_dir, 'data', config['fpattern_captions_val'])
    fpattern_captions_test = os.path.join(main_dir, 'data', config['fpattern_captions_test'])
    fpaths_captions_train = glob.glob(fpattern_captions_train)
    fpaths_captions_val = glob.glob(fpattern_captions_val)
    fpaths_captions_test = glob.glob(fpattern_captions_test)

    fpattern_caption_vectors_train = os.path.join(
        main_dir, 'preprocess', config['fpattern_captions_train'] + '.pt')
    fpattern_caption_vectors_val = os.path.join(
        main_dir, 'preprocess', config['fpattern_captions_val'] + '.pt')
    fpattern_caption_vectors_test = os.path.join(
        main_dir, 'preprocess', config['fpattern_captions_test'] + '.pt')
    fpaths_caption_vectors_train = glob.glob(fpattern_caption_vectors_train)
    fpaths_caption_vectors_val = glob.glob(fpattern_caption_vectors_val)
    fpaths_caption_vectors_test = glob.glob(fpattern_caption_vectors_test)
    fpath_vocab = os.path.join(main_dir, 'preprocess', config['fname_vocab'])
        
    # train
    fpath_im_features_train = os.path.join(
        main_dir, 'preprocess', config['fname_image_features_train'])
    fpath_im_features_val = os.path.join(
        main_dir, 'preprocess', config['fname_image_features_val'])
    fpath_im_features_test = os.path.join(
        main_dir, 'preprocess', config['fname_image_features_test'])
    fpath_losses = os.path.join(main_dir, 'train', model_dir, 'losses.pt')
    fpath_bleu_scores = os.path.join(main_dir, 'train', model_dir, 'bleu_scores.pt')
    fpath_model = os.path.join(main_dir, 'train', model_dir, f'model.%d.pt')
    fpath_best_model = os.path.join(main_dir, 'train', model_dir, f'model_best.pt')

    # predict
    fpath_predictions_val = os.path.join(main_dir, 'predict', model_dir, 'predictions_val.txt')
    fpath_predictions_test = os.path.join(main_dir, 'predict', model_dir, 'predictions_test.txt')
    fpath_predictions_train = os.path.join(main_dir, 'predict', model_dir, 'predictions_train.txt')
    
    # report
    fpath_plot_epoch_loss = os.path.join(main_dir, 'report', model_dir, 'epoch_losses.png')
    fpath_plot_batch_loss = os.path.join(main_dir, 'report', model_dir, 'batch_losses.png')
    fpath_plot_bleu = os.path.join(main_dir, 'report', model_dir, 'bleu.png')
    fpath_bleu_val = os.path.join(main_dir, 'report', model_dir, 'BLEU_val.txt')
    fpath_bleu_test = os.path.join(main_dir, 'report', model_dir, 'BLEU_test.txt')
    fpath_bleu_train = os.path.join(main_dir, 'report', model_dir, 'BLEU_train.txt')
    fpath_bleu_human_test = os.path.join(main_dir, 'report', model_dir, 'human_BLEU_test.txt')
    
    # statistics
    fpath_word_frequencies = os.path.join(main_dir, 'statistics', 'word_frequencies.png')
    fpath_sentence_lengths = os.path.join(main_dir, 'statistics', 'sentence_lengths.png')

    # logging
    fpath_train_out = None
    fpath_predict_out = None
    fpath_report_out = None
    if config.get('fname_out'):
        fpath_train_out = os.path.join(main_dir, 'train', model_dir, config['fname_out'])
        fpath_predict_out = os.path.join(main_dir, 'predict', model_dir, config['fname_out'])
        fpath_report_out = os.path.join(main_dir, 'report', model_dir, config['fname_out'])
    
    return {
        'captions_train' : fpaths_captions_train,
        'captions_val' : fpaths_captions_val,
        'captions_test' : fpaths_captions_test,
        'caption_vectors_train' : fpaths_caption_vectors_train,
        'caption_vectors_val' : fpaths_caption_vectors_val,
        'caption_vectors_test' : fpaths_caption_vectors_test,
        'vocab' : fpath_vocab,
        'image_features_train' : fpath_im_features_train,
        'image_features_val': fpath_im_features_val,
        'image_features_test': fpath_im_features_test,
        
        'losses' : fpath_losses,
        'bleu_scores' : fpath_bleu_scores,
        'model' : fpath_model,
        'best_model' : fpath_best_model,
        'train_out' : fpath_train_out,
        
        'predictions_test' : fpath_predictions_test,
        'predictions_val' : fpath_predictions_val,
        'predictions_train' : fpath_predictions_train,
        'predict_out' : fpath_predict_out,
        
        'plot_epoch_loss' : fpath_plot_epoch_loss,
        'plot_batch_loss' : fpath_plot_batch_loss,
        'plot_bleu' : fpath_plot_bleu,
        'bleu_val' : fpath_bleu_val,
        'bleu_test' : fpath_bleu_test,
        'bleu_train' : fpath_bleu_train,
        'bleu_human_test' : fpath_bleu_human_test,
        'report_out' : fpath_report_out,
        
        'word_frequencies' : fpath_word_frequencies,
        'sentence_lengths' : fpath_sentence_lengths
    }

def replace_env_vars(d):
    tmpdir = os.environ.get('TMPDIR', '') 
    return {k : v.replace('$TMPDIR', tmpdir) for k, v in d.items()}

    # train
    #### OLD
#    fpath_image_split_train = os.path.join(main_dir, "image_splits", "train_images.txt")
#    fpath_image_split_val = os.path.join(main_dir, "image_splits", "val_images.txt")
#    fpath_image_split_test = os.path.join(main_dir, "image_splits", "val_images.txt")
#    dir_image_features = os.path.join(output_dir_preprocess, 'resnet18_avgpool')
#    fpaths_image_features_train = fpaths_image_split(dir_image_features, fpath_image_split_train, True)
#    fpaths_image_features_val = fpaths_image_split(dir_image_features, fpath_image_split_val, True)
#    fpaths_image_features_test = fpaths_image_split(dir_image_features, fpath_image_split_test, True)

#def fpaths_image_split(dir_images, fpath_image_split, is_encoded = False): 
#    fnames_image_split = read_lines(fpath_image_split)
#    added_ext = ".pt" if is_encoded else ""
#    return [os.path.join(dir_images, f'{fname}{added_ext}') for fname in fnames_image_split]


