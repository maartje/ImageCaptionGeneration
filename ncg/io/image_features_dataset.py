import torch
from torch.utils import data
import math

class ImageFeaturesDataset(data.Dataset):

    def __init__(self, fpaths_image_features):
        super(ImageFeaturesDataset, self).__init__()
        self.fpaths_image_features = fpaths_image_features

    def __len__(self):
        return len(self.fpaths_image_features)

    def __getitem__(self, index):
        fpath = self.fpaths_image_features[index]
        image_features = torch.load(fpath)
        return image_features
        

