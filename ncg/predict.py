import torch
from ncg.nn.train_model import predict as model_predict
from torch.utils import data
from ncg.io.image_features_dataset import ImageFeaturesDataset

def predict(fpaths_image_features, fpath_decoder, fpath_vocab,
            fpath_save_predictions, max_length = 100, dl_params = {}):

    # create models and data
    decoder = torch.load(fpath_decoder)
    text_mapper = torch.load(fpath_vocab)
    data_set = ImageFeaturesDataset(fpaths_image_features)
    data_loader = data.DataLoader(data_set)

    # predict captions
    predicted_sentences = []
    SOS_index = text_mapper.SOS_index()
    EOS_index = text_mapper.EOS_index()
    for image_encoding in data_loader:
        predicted_indices = model_predict(
            decoder, image_encoding, SOS_index, EOS_index, max_length)
        predicted_sentence = text_mapper.indices2sentence(predicted_indices)
        predicted_sentences.append(predicted_sentence)

    open(fpath_save_predictions, "w").write('\n'.join(predicted_sentences))
