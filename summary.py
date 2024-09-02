from torchinfo import summary
from model.deeplabv3_torchvision import Create_DeepLabV3_ResNet101, Create_DeepLabV3_ResNet18
from model.deeplabv3_custom import get_deeplabv3
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
    col_names=["input_size", "output_size", "num_params"],
    col_width=20,
    device=args.device,
    row_settings=["var_names"])

if __name__ == "__main__":
    if args.model == "deeplabv3_resnet101":
        model = Create_DeepLabV3_ResNet101(12, args)
        show_summary(model)
    elif args.model == "deeplabv3_resnet18":
        model = Create_DeepLabV3_ResNet18(12, args)
        show_summary(model)
    elif args.model == "deeplabv3_cirkd":
        model = get_deeplabv3(num_classes=12, backbone="resnet101", args=args)
        show_summary(model)
    elif args.model == "enet":
        model = Create_ENet(12)
        show_summary(model)
    else:
        raise ValueError("Unknown model")