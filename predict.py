from parse_config import get_configuration

from ncg.predict import predict

from filepaths import fpaths_image_split
from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist

def predict_image_descriptions(config):
    fpaths_images = fpaths_image_split(
        config['dir_images'], config['fpath_image_split'], is_encoded = False)
    fpath_decoder = config['fpath_decoder']
    fpath_vocab = config['fpath_vocab']
    fpath_save_predictions = config['fpath_save_predictions']
    encoder_model = config['encoder_model']
    encoder_layer = config['encoder_layer']
    max_length = config['max_length']
    dl_params = config['dl_params']

    check_files_not_exist([fpath_save_predictions]) 
    ensure_paths_exist([fpath_save_predictions])

    predict(fpaths_images, fpath_decoder, fpath_vocab,
            fpath_save_predictions, encoder_model, encoder_layer,
            max_length, dl_params)
        
def main():
    config = get_configuration('predict', 
                               description = 'Predict image descriptions for train or validation set.')
    predict_image_descriptions(config)


if __name__ == "__main__":
    main()
     
      

