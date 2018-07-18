import torch
from ncg.nn.predict import predict as model_predict
from torch.utils import data
from ncg.io.image_features_dataset import ImageFeaturesDataset

def predict(fpath_imfeats, fpath_decoder, fpath_vocab,
            fpath_save_predictions, max_length = 50, dl_params = {'batch_size' : 256}):

    # create models and data
    decoder = torch.load(fpath_decoder)
    text_mapper = torch.load(fpath_vocab)
    data_set = ImageFeaturesDataset(fpath_imfeats)
    data_loader = data.DataLoader(data_set, **dl_params)

    # predict captions
    SOS_index = text_mapper.SOS_index()
    EOS_index = text_mapper.EOS_index()
    predicted_indices = model_predict(decoder, data_loader, SOS_index, max_length)
    predicted_sentences = [text_mapper.indices2sentence(s) for s in predicted_indices]
    
    # save to disk
    open(fpath_save_predictions, "w").write('\n'.join(predicted_sentences))
