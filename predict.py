from parse_config import get_configuration

from ncg.predict import predict

from filepaths import fpaths_image_split
from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist

def predict_image_descriptions(config, filepaths):

    check_files_exist([filepaths['image_features_val'], filepaths['image_features_test']])
    ensure_paths_exist([filepaths['predictions_val']])
    
    # validation
    predict(filepaths['image_features_val'], filepaths['model'], filepaths['vocab'],
            filepaths['predictions_val'], config['max_length'], config['dl_params'])

    # test
    predict(filepaths['image_features_test'], filepaths['model'], filepaths['vocab'],
            filepaths['predictions_test'], config['max_length'], config['dl_params'])

    # train
    predict(filepaths['image_features_train'], filepaths['model'], filepaths['vocab'],
            filepaths['predictions_train'], config['max_length'], dl_params)
        
def main():
    config, filepaths = get_configuration('predict', 
                               description = 'Predict image descriptions for train or validation set.')
    predict_image_descriptions(config, filepaths)


if __name__ == "__main__":
    main()
     
      

