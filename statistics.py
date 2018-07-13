import glob

from parse_config import get_configuration

from helpers import ensure_paths_exist
from ncg.show_stats import info_and_statistics

def show_info_and_statistics(config, filepaths):    
    ensure_paths_exist([filepaths['word_frequencies'], filepaths['sentence_lengths']])
    
    info_and_statistics(
        filepaths['vocab'], filepaths['captions_train'], filepaths['model'],
        filepaths['word_frequencies'], filepaths['sentence_lengths'])        
        
def main():
    config, filepaths = get_configuration('statistics', 
                               description = 'Statistics and info about dataset and neural models.')
    show_info_and_statistics(config, filepaths)

if __name__ == "__main__":
    main()
     
      

