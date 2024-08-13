import torch
from torchinfo import summary
from model.deeplabv3 import Create_DeepLabV3
from model.enet import Create_ENet
from utils.utils import merge_args_with_config

import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
args = merge_args_with_config(args, config)

def show_summary(model):
    summary(model=model,
            input_size=(1, 3, args.width, args.height),
            col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"],
            col_width=20,
            device=args.device,
            row_settings=["var_names"],
            verbose=1)

if __name__ == "__main__":
    if args.model == "deeplabv3":
        model = Create_DeepLabV3(11, args)
        show_summary(model)
    elif args.model == "enet":
        model = Create_ENet(11)
        show_summary(model)
    elif args.model == "all":
        model = Create_DeepLabV3(11, args)
        show_summary(model)

        model = Create_ENet(11)
        show_summary(model)
    else:
        raise ValueError("Unknown model")