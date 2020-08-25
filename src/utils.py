from PIL import Image
from io import BytesIO
import numpy as np

import torch
from torchvision import transforms


def load_image(img_path, max_size, shape=None):
    '''
        Load and convert the image,
        make sure that it is less than max_size pixels in the x-y dims,
        resize and normalize it.
    '''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

    # Image without the alpha channel.
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# Helper function for un-normalizing an image
# And converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = (image * np.array((0.229, 0.224, 0.225))
             + np.array((0.485, 0.456, 0.406)))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """
        Run an image forward through a model and get the features for
        a set of layers.
        Default layers are for VGGNet matching Gatys et al (2016)
    """

    # Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation
                  '28': 'conv5_1'}

    features = {}
    x = image
    # Extracting the wanted features for a given image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """
        Calculate the Gram Matrix of a given tensor
    """
    # Get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    # Reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    # Calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram
