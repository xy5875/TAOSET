import argparse
from alexnet import AlexNet_running
from vgg16net import Vgg16Net_running
from resnet50 import ResNet50_running

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select training model and whether execute mem_recycle.")
    parser.add_argument("--network", type=str, default='alexnet', help="select model")
    parser.add_argument("--empty_cache", type=str, default='False', help="pass mem_recycle value")

    args = parser.parse_args()

    if args.network == 'alexnet':
        AlexNet_running(args)
    elif args.network == 'vgg16net':
        Vgg16Net_running(args)
    elif args.network == 'resnet50':
        ResNet50_running(args)
    else:
        print("Please provide correct value.")