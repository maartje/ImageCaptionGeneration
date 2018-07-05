import torch
from torch.utils import data
import math

class EmbeddingDescriptionDataset(data.Dataset):

    def __init__(self, fpaths_embeddings, fpaths_description_vectors):
        super(EmbeddingDescriptionDataset, self).__init__()
        self.description_lists = [torch.load(fpath) for fpath in fpaths_description_vectors]
        self.fpaths_embeddings = fpaths_embeddings

    def __len__(self):
        return len(self.description_lists) * len(self.fpaths_embeddings)

    def __getitem__(self, index):
        nr_of_embeddings = len(self.fpaths_embeddings)
        index_embedding = index % nr_of_embeddings
        index_description_list = math.floor(index / nr_of_embeddings)
        embedding = torch.load(self.fpaths_embeddings[index_embedding])
        description_vector = self.description_lists[index_description_list][index_embedding]
        return embedding, torch.LongTensor(description_vector)
        
class EmbeddingDescriptionGroupedDataset(data.Dataset):

    def __init__(self, fpaths_embeddings, fpaths_description_vectors):
        super(EmbeddingDescriptionGroupedDataset, self).__init__()
        self.description_lists = [torch.load(fpath) for fpath in fpaths_description_vectors]
        self.fpaths_embeddings = fpaths_embeddings
        self.cached_embedding = None
        self.cached_index_embedding = -1

    def __len__(self):
        return len(self.description_lists) * len(self.fpaths_embeddings)

    def __getitem__(self, index):
        descriptions_per_image = len(self.description_lists)
        index_description_list = index % descriptions_per_image
        index_embedding = math.floor(index / descriptions_per_image)
        if self.cached_index_embedding == index_embedding:
            embedding = self.cached_embedding
        else:
            embedding = torch.load(self.fpaths_embeddings[index_embedding])
            self.cached_index_embedding = index_embedding
            self.cached_embedding = embedding
        description_vector = self.description_lists[index_description_list][index_embedding]
        return embedding, torch.LongTensor(description_vector)


