import glob

from helpers import check_files_exist, check_files_not_exist, ensure_paths_exist
from parse_config import get_configuration

from ncg.report import plot_losses, calculate_metrics, compare_with_human_performance

def report(config):
    # losses
    fpath_losses = config.get('fpath_losses')
    fpath_epoch_loss = config.get('fpath_plot_epoch_loss')
    fpath_batch_loss = config.get('fpath_plot_batch_loss')
    try:
        report_losses(fpath_losses, fpath_epoch_loss, fpath_batch_loss)
        print (f'losses reported in {fpath_epoch_loss} and/or {fpath_batch_loss}')
    except (FileNotFoundError, FileExistsError):
        print('Error occurred, no loss info reported')
        
    # validation metrics
    fpattern_captions = config.get('fpattern_captions_val', '')
    fpath_predicted = config.get('fpath_predictions_val')
    fpath_save_bleu = config.get('fpath_save_bleu_val')
    fpath_save_human_comparison = config.get('fpath_save_human_comparison_val')
    try:
        report_metrics(fpattern_captions, fpath_predicted, fpath_save_bleu, 
                                    fpath_save_human_comparison)
        print (f'validation metrics reported in {fpath_save_bleu} and/or {fpath_save_human_comparison}')
    except (FileNotFoundError, FileExistsError):
        print('Error occurred, no validation metrics reported')

def report_losses(fpath_losses, fpath_epoch_loss, fpath_batch_loss):
    if fpath_epoch_loss or fpath_batch_loss:
        check_files_not_exist([fpath_epoch_loss, fpath_batch_loss])
        ensure_paths_exist([fpath_epoch_loss, fpath_batch_loss])
        plot_losses(
            fpath_losses, 
            fpath_epoch_loss, 
            fpath_batch_loss)

def report_metrics(fpattern_captions, fpath_predicted, fpath_save_bleu, 
                                fpath_save_human_comparison):
    fpaths_references = glob.glob(fpattern_captions)
    if fpath_save_bleu:
        if not len(fpaths_references):
            print('Calculating a bleu score requires at least one annotated file.')
            return
        check_files_not_exist([fpath_save_bleu])
        ensure_paths_exist([fpath_save_bleu])
        calculate_metrics(fpaths_references, fpath_predicted, fpath_save_bleu)
    if fpath_save_human_comparison:
        if len(fpaths_references) < 2:
            print('Comparison with human score requires at least two annotated files.')
            return
        check_files_not_exist([fpath_save_human_comparison])
        ensure_paths_exist([fpath_save_human_comparison])
        compare_with_human_performance(fpaths_references, fpath_predicted,   
                                       fpath_save_human_comparison)   

def main():
    config = get_configuration('report', 
                               description = 'Plots of losses and evaluation metrics.')
    report(config)


if __name__ == "__main__":
    main()
     
      

