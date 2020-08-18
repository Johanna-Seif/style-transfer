import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import models
# import requests

from util import load_image, im_convert, get_features, gram_matrix


# Extracting and fixing features of pretrained VGG19
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)  # Prevent any changes

# Move the model to GPU, if available
# TODO uncomment changing of device when running the training
# TODO add it as an option with error gestion if gpu asked but not avalaible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("The device used is: {}".format(device))
vgg.to(device)

# load in content and style image
content = load_image('../images/janelle.png').to(device)
# Resize style to match content, makes code easier
style = load_image('../images/ben_passmore.jpg',
                        shape=content.shape[-2:]).to(device)

# # display the images
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# # content and style ims side-by-side
# ax1.imshow(im_convert(content))
# ax2.imshow(im_convert(style))
# plt.show()

# get content and style features only once before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# start off with the target as a copy of our *content* image
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta

# for displaying the target image, intermittently
show_every = 20

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 20  # decide how many iterations to update your image (5000)

for ii in range(1, steps+1):

    # get the features from your target image
    target_features = get_features(target, vgg)

    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()

# TODO write image
# breakpoint()
