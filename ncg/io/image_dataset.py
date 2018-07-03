from torch.utils import data
from PIL import Image
from torchvision import transforms

class ImageDataset(data.Dataset):

    def __init__(self, fpaths_images):
        super(ImageDataset, self).__init__()
        self.fpaths_images = fpaths_images
        self.transform = transforms.Compose([ #TODO: pass as arg, size in config, figure out mean, std
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.fpaths_images)

    def __getitem__(self, index):
        fpath = self.fpaths_images[index]
        img = Image.open(fpath)
        return (self.transform(img), fpath)
        
        

