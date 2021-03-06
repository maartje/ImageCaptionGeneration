import torch
import random
from torch.utils import data
import numpy
from PIL import Image

def generate_random_image(width = 600, height = 400):
    imarray = numpy.random.rand(width, height, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    return img

def generate_random_encoding(encoding_size):
    return 2*torch.rand(encoding_size)

def generate_random_caption(vocab_size, min_length = 5, max_length = 10):
    # TODO: list of random ints, float tensor in MockDS
    caption_size = random.randint(min_length, max_length)
    return [0] + random.sample(range(2, vocab_size), caption_size) + [1]

def generate_random_training_pair(encoding_size, vocab_size, min_length = 5, max_length = 10):
    caption = torch.LongTensor(
        generate_random_caption(vocab_size, min_length, max_length)).view(1,-1)
    encoding = generate_random_encoding(encoding_size).view(1,-1)
    return encoding, caption

class MockEmbeddingDescriptionDataset(data.Dataset):

    encoding_size = 64 
    vocab_size = 100
    
    def __init__(self, fpaths_image_encodings, fpaths_captions):
        super(MockEmbeddingDescriptionDataset, self).__init__()
        
        def random_encoding_words_pair():
            return (
                generate_random_encoding(type(self).encoding_size), 
                generate_random_caption(type(self).vocab_size, 10, 15)[1:-1]
            )

        def sample_train_pairs(encoding_words_pairs):
            return [
                (e, torch.LongTensor(
                    [0] + random.sample(w, len(w) - 5) + [1] )) for e, w in encoding_words_pairs
            ]
        random.seed(9001)
        torch.manual_seed(9001)
        encoding_words_pairs = [random_encoding_words_pair() for _ in fpaths_image_encodings]
        training_pairs = [sample_train_pairs(encoding_words_pairs) for _ in fpaths_captions]
        self.training_pairs = [item for sublist in training_pairs for item in sublist]
        
    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, index):
        return self.training_pairs[index]
        

