import torch
from torch.utils import data
import math

class ImageCaptionDataset(data.Dataset):

    def __init__(self, fpaths_image_encodings, fpaths_caption_vectors):
        super(ImageCaptionDataset, self).__init__()
        self.caption_lists = [torch.load(fpath) for fpath in fpaths_caption_vectors]
        self.fpaths_image_encodings = fpaths_image_encodings

    def __len__(self):
        return len(self.caption_lists) * len(self.caption_lists[0])

    def __getitem__(self, index):
        nr_of_images = len(self.caption_lists[0])
        index_image = index % nr_of_images
        index_caption_list = math.floor(index / nr_of_images)
        image_encoding = torch.load(self.fpaths_image_encodings[index_image])
        caption_vector = self.caption_lists[index_caption_list][index_image]
        return image_encoding, torch.LongTensor(caption_vector)
        

