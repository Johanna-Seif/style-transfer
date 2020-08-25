import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import models

from utils import load_image, im_convert, get_features, gram_matrix


def style_transfer(device, content_image_path, style_image_path,
                   output_image_path, max_size, content_weight,
                   style_weight, steps, show_every,
                   conv_style_weights):
    '''
        Style transfer on a content image.
    '''
    # Extracting and fixing features of pretrained VGG19
    vgg = models.vgg19(pretrained=True).features
    # Preventing any change on the weights of the original model
    for param in vgg.parameters():
        param.requires_grad_(False)

    # Move the model to the device requested
    torch.cuda.empty_cache()
    try:
        device = torch.device(device)
        print("The device used is: {}".format(device))
    except BaseException:
        print("Error, the required device is not available")
    vgg.to(device)

    # Load in content and style image
    content = load_image(content_image_path, max_size).to(device)
    # Resize style to match content
    style = load_image(style_image_path, max_size,
                       shape=content.shape[-2:]).to(device)

    # Content and style features before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer])
                   for layer in style_features}

    # Create a third "target" image and prep it for change
    target = content.clone().requires_grad_(True).to(device)

    # Iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)

    for ii in range(1, steps+1):

        # Get the features from your target image
        target_features = get_features(target, vgg)

        # The content loss
        content_loss = torch.mean((target_features['conv4_2'] -
                                   content_features['conv4_2'])**2)

        # The style loss
        style_loss = 0
        # Add loss for each layer's gram matrix loss
        for layer in conv_style_weights:
            # Get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # Get the "style" style representation
            style_gram = style_grams[layer]
            # The style loss for one layer, weighted appropriately
            layer_style_loss = (conv_style_weights[layer]
                                * torch.mean((target_gram - style_gram)**2))
            # Add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # Calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Display intermediate images and print the loss
        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.axis('off')
            plt.savefig(output_image_path + "_{}.jpg".format(ii // show_every),
                        bbox_inches='tight', pad_inches=0)
            # plt.show()
    plt.imshow(im_convert(target))
    plt.axis('off')
    plt.savefig(output_image_path + "_final.jpg",
                bbox_inches='tight', pad_inches=0)
    torch.cuda.empty_cache()
