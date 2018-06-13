from torchvision import models, transforms

def get_image_encoder(model_name, layer_name):
    encoder_models = {
        'resnet18': models.resnet18,
        'resnet152': models.resnet152,
        'vgg16': models.vgg16
    }
    model = encoder_models[model_name](pretrained=True) 
    layer = model._modules.get(layer_name)
    model.eval()
    return model, layer

def encode_image(model, layer, img):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ])
    t_img = transform(img).unsqueeze(0)
    return _embed(model, layer, t_img)

def _embed(model, layer, t_img):
    embedding = [] # stores the output for the given layer
    def copy_data(m, i, o):
        embedding.append(o.clone())
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()    
    return embedding[0].squeeze()


