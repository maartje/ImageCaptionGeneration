from torchvision import models, transforms
import torch

class ImageEncoder:

    def __init__(self, model_name, layer_name, encoder_models = {}):
        self.model_name = model_name
        self.layer_name = layer_name
        self.model = None
        self.layer = None
        self.encoder_models = {
            'resnet18': models.resnet18,
            'resnet152': models.resnet152,
            'vgg16': models.vgg16
        }
        self.encoder_models.update(encoder_models)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225]),
        ])


    def load_model(self):
        self.model = self.encoder_models[self.model_name](pretrained=True) 
        self.layer = self.model._modules.get(self.layer_name)
        self.model.eval()

    def encode_image(self, img):
        t_img = self.transform(img).unsqueeze(0)
        return self._embed(t_img)

    def encode_images(self, imgages):
        t_imgages = torch.stack([self.transform(img) for img in imgages])
        return self._embed(t_imgages)

    def _embed(self, t_img):
        embedding = [] # stores the output for the given layer
        def copy_data(m, i, o):
            embedding.append(o.clone())
        h = self.layer.register_forward_hook(copy_data)
        with torch.no_grad():
            self.model(t_img)
        h.remove()    
        return embedding[0].squeeze()


