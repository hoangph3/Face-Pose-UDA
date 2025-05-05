import argparse
import torch

def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Pose UDA Trainer')
    # parse args
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args