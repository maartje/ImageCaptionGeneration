from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist
from parse_config import get_configuration

from ncg.report import plot_losses, calculate_metrics, compare_with_human_performance

def report(config, filepaths):
    ensure_paths_exist([filepaths['plot_epoch_loss']])
    plot_losses(filepaths['losses'], filepaths['plot_epoch_loss'], filepaths['plot_batch_loss'])

    calculate_metrics(
        filepaths['captions_val'], filepaths['predictions_val'], filepaths['bleu_val'])
    calculate_metrics(
        filepaths['captions_train'], filepaths['predictions_train'], filepaths['bleu_train'])
#    calculate_metrics(
#        filepaths['captions_test'], filepaths['predictions_test'], filepaths['bleu_test'])

#    compare_with_human_performance(
#        filepaths['captions_test'], filepaths['predictions_test'], filepaths['bleu_human_test'])


def main():
    config, filepaths = get_configuration('report', 
                               description = 'Plots of losses and evaluation metrics.')
    report(config, filepaths)


if __name__ == "__main__":
    main()
     
      

