import torch
from ncg.nn.train_model import predict as model_predict
from torch.utils import data
from ncg.data_processing.image_encoder import ImageEncoder
from ncg.io.image_dataset import ImageDataset

def predict(fpaths_images, fpath_decoder, fpath_vocab,
            fpath_save_predictions, encoder_model, encoder_layer,
            max_length = 100, dl_params = {}):

    # create models and data
    image_encoder = ImageEncoder(encoder_model, encoder_layer)
    image_encoder.load_model()
    decoder = torch.load(fpath_decoder)
    text_mapper = torch.load(fpath_vocab)
    data_set = ImageDataset(fpaths_images)
    data_loader = data.DataLoader(data_set)

    # predict captions
    predicted_sentences = []
    SOS_index = text_mapper.token2index(text_mapper.SOS)
    EOS_index = text_mapper.token2index(text_mapper.EOS)
    for img, _ in data_loader:
        image_encoding = image_encoder.encode(img)
        predicted_indices = model_predict(
            decoder, image_encoding, SOS_index, EOS_index, max_length)
        predicted_sentence = text_mapper.indices2sentence(predicted_indices)
        predicted_sentences.append(predicted_sentence)

    open(fpath_save_predictions, "w").write('\n'.join(predicted_sentences))
