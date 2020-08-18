import argparse


parser = argparse.ArgumentParser(description='Style Transfer Program')

parser.add_argument('--device', metavar='DEVICE',
                    type=str, default='cpu',
                    help='cuda if you want to run of gpu. Default is cpu')


if __name__ == '__main__':
    print("\n end of main \n")
