import torch
from torch.utils import data
import tables

class ImageFeaturesDataset(data.Dataset):

    def __init__(self, fpath_image_features):
        super(ImageFeaturesDataset, self).__init__()
        self.fpath_image_features = fpath_image_features
        self.h5file = tables.open_file(fpath_image_features, mode="r")
        self.global_feats = self.h5file.root.global_feats

    def __len__(self):
        return len(self.global_feats)

    def __getitem__(self, index):
        return torch.FloatTensor(self.global_feats[index])
        

