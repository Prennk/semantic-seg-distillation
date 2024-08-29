import torch.nn as nn
import torch.optim as optim
import yaml
from argparse import ArgumentParser
from timeit import default_timer as timer
from torchinfo import summary

from utils import utils, loops, metrics, transforms as ext_transforms, data_utils
from model.enet import Create_ENet
from model.deeplabv3_torchvision import Create_DeepLabV3
from model.deeplabv3_custom import get_deeplabv3

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
args = utils.merge_args_with_config(args, config)

args.pretrained = False
args.freeze = False

def test(model, test_loader, class_weights, class_encoding):
    print("\nStart testing...")

    num_classes = len(class_encoding)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric_iou = metrics.IoU(num_classes, ignore_index=ignore_index)
    metric_pa = metrics.PixelAccuracy(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = loops.Test(model, test_loader, criterion, metric_iou, metric_pa, args.device)

    print("Running test dataset...")

    loss, (iou, miou), (pa, mpa), test_time = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))
    class_pa = dict(zip(class_encoding.keys(), pa))

    print("Result => Avg. loss: {0:.4f} | mIoU: {1:.4f} | mPA: {2:.4f} | time elapsed: {3:.3f}".format(loss, miou, mpa, test_time))

    for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
        print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

def main(mode="train"):
    print("starting inference...")
    from data.camvid import CamVid as dataset

    loaders, w_class, class_encoding = data_utils.load_dataset(dataset, args)
    train_loader, val_loader, test_loader = loaders
    num_classes = len(class_encoding)
    
    if mode in ["train, test"]:
        if args.model == 'enet':
            model = Create_ENet(num_classes).to(args.device)
        elif args.model == "deeplabv3_torch":
            model = Create_DeepLabV3(num_classes, args).to(args.device)
        elif args.model == "deeplabv3_cirkd":
            model = get_deeplabv3(num_classes=num_classes, backbone="resnet101", pretrained=True, args=args).to(args.device)
        else:
            raise TypeError('Invalid model name. Available models are enet and deeplabv3_resnet101')
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                    args.name)[0]
        
    elif mode == "distill":
        model = Create_DeepLabV3(num_classes, args).to(args.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                            args.name)[0]
    else:
        print(f"Invalid {args.mode}")

    test(model, test_loader, w_class, class_encoding)

if __name__ == "__main__":
    main()