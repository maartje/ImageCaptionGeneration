import torch
import ncg.statistics.text_stats as textinfo
import ncg.statistics.model_stats as modelinfo
from ncg.data_processing.image_encoder import ImageEncoder

def mj():
    print('mj')

def info_and_statistics(
        fpath_vocab, fpaths_captions, fpath_decoder, fpath_plot_word_frequencies, 
        fpath_plot_sentencelengths, encoder_model, encoder_layer):
    info_and_statistics_text(fpath_vocab, fpaths_captions, 
                             fpath_plot_word_frequencies, fpath_plot_sentencelengths)
    info_and_statistics_model(encoder_model, encoder_layer, fpath_decoder)
    info_and_statistics_images()

def info_and_statistics_text(fpath_vocab, fpaths_captions, fpath_plot_word_frequencies, 
                             fpath_plot_sentencelengths):
    vocab = torch.load(fpath_vocab).vocab
    textinfo.printWordStats(vocab)
    textinfo.plotWordFrequencies(vocab, fpath_plot_word_frequencies)
    textinfo.plotSentenceLengthFrequencies(fpaths_captions, fpath_plot_sentencelengths)
    
def info_and_statistics_model(encoder_model, encoder_layer, fpath_decoder):
    img_encoder = ImageEncoder(encoder_model, encoder_layer)
    img_encoder.load_model()
    modelinfo.printEncoderInfo(img_encoder.model)

    decoder = torch.load(fpath_decoder)
    modelinfo.printDecoderInfo(decoder)

def info_and_statistics_images():
    print("TODO: stdev and mean for channels, useful for normalization see: ncg.io.image_dataset")
    

