import argparse

def argument_parser():
    '''
        Parse the arguments for the program
    '''
    parser = argparse.ArgumentParser(description='Style Transfer Program')

    parser.add_argument(
        '--device', metavar='DEVICE',
        type=str, default='cpu',
        choices=['cpu','cuda'],
        help='Specify cuda if you want to run on gpu. Default is cpu')
    parser.add_argument(
        '--content_image_path', metavar='CONTENT_IMAGE_PATH',
        type=str, required=True,
        help='Path to the content image')
    parser.add_argument(
        '--style_image_path', metavar='STYLE_IMAGE_PATH',
        type=str, required=True,
        help='Path to the style image')
    parser.add_argument(
        '--output_image_path', metavar='OUTPUT_IMAGE_PATH',
        type=str, default='./images',
        help='Path to the save the output image. Default ./images')
    parser.add_argument(
        '--max_size', metavar='MAX_SIZE',
        type=int, default=400,
        help='Max size for the input and output images. Default 400.')
    parser.add_argument(
        '--content_weight', metavar='CONTENT_WEIGHT',
        type=int, default=1,
        help='Content image weight. Default 1.')
    parser.add_argument(
        '--style_weight', metavar='STYLE_WEIGHT',
        type=int, default=1e6,
        help='Style image weight. Default 1e6.')
    parser.add_argument(
        '--steps', metavar='STEPS',
        type=int, default=2000,
        help='Number of update steps. Default 2000.')
    parser.add_argument(
        '--show_every', metavar='SHOW_EVERY',
        type=int,
        help='Display the target image every SHOW_EVERY steps. Optional.')
    parser.add_argument(
        '--conv1_1', metavar='CONV1_1',
        type=int, default=1,
        help='First convolutional layer weight. Default 1.')
    parser.add_argument(
        '--conv1_2', metavar='CONV1_2',
        type=int, default=1,
        help='Second convolutional layer weight. Default 0.75.')
    parser.add_argument(
        '--conv1_3', metavar='CONV1_3',
        type=int, default=1,
        help='Third convolutional layer weight. Default 0.2.')
    parser.add_argument(
        '--conv1_4', metavar='CONV1_4',
        type=int, default=1,
        help='Fourth convolutional layer weight. Default 0.2.')
    parser.add_argument(
        '--conv1_5', metavar='CONV1_5',
        type=int, default=1,
        help='Fifth convolutional layer weight. Default 0.2.')

    return parser

if __name__ == '__main__':

    parser = argument_parser()
    args = parser.parse_args()

    print(args.show_every)
    if show_every == None:
        show_every = steps + 1

    # Weights for each style layer
    # Weighting earlier layers more will result in *larger* style artifacts
    style_weights = {'conv1_1': args.conv1_1,
                     'conv2_1': args.conv1_2,
                     'conv3_1': args.conv1_3,
                     'conv4_1': args.conv1_4,
                     'conv5_1': args.conv1_5}

    print("\n end of main \n")
