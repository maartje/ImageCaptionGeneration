import glob

from parse_config import get_configuration

from helpers import ensure_paths_exist
from ncg.show_stats import info_and_statistics

def show_info_and_statistics(config):
    fpath_vocab = config['fpath_vocab'] 
    fpath_decoder = config['fpath_decoder'] 
    fpath_plot_word_frequencies = config['fpath_plot_word_frequencies'] 
    fpath_plot_sentencelengths = config['fpath_plot_sentencelengths']
    encoder_model = config['encoder_model']
    encoder_layer = config['encoder_layer']
    fpattern_captions_train = config['fpattern_captions_train']
    fpaths_captions = glob.glob(fpattern_captions_train)
    
    ensure_paths_exist([fpath_plot_word_frequencies, fpath_plot_sentencelengths])
    
    info_and_statistics(
        fpath_vocab, fpaths_captions, fpath_decoder, fpath_plot_word_frequencies,   
        fpath_plot_sentencelengths, encoder_model, encoder_layer)        
        
def main():
    config = get_configuration('statistics', 
                               description = 'Statistics and info about dataset and neural models.')
    show_info_and_statistics(config)

if __name__ == "__main__":
    main()
     
      

