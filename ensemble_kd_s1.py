import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import yaml
from argparse import ArgumentParser
import os
import numpy as np
import random
from collections import OrderedDict
from timeit import default_timer as timer
from torchinfo import summary

from utils import utils, loops, loop_ensemble, metrics, transforms as ext_transforms, data_utils
from model.enet import Create_ENet
from model.deeplabv3_resnet101 import Create_DeepLabV3_ResNet101
from model.deeplabv3_mobilenetv3 import Create_DeepLabV3_MobileNetV3
from model.deeplabv3_custom import get_deeplabv3
from distiller.kd import CriterionKD
from distiller.vid import VIDLoss
from distiller.fsp import FSP
from distiller.dtkd import DTKD
from inference import main as inference

# get config from config.yaml
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
args = utils.merge_args_with_config(args, config)

# print("-" * 70)
# print("| Config" + " " * 61 + "|")
# print("-" * 70)
# for key, value in vars(args).items():
#     print(f"| {key}: {value}" + " " * (67 - len(f"{key}: {value}")) + "|")
# print("-" * 70)
# print()

# seed everything
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

def train_ensemble_kd(train_loader, val_loader, class_weights, class_encoding, args):
    num_classes = len(class_encoding)
    print(f"num classes: {num_classes}")

    # create teacher
    print(f"\nLoading teacher model: deeplabv3 from {args.teacher_path}...")
    t_model_1 = Create_DeepLabV3_ResNet101(num_classes, args, layers_to_hook=args.teacher_layers).to(args.device)
    t_model_2 = Create_DeepLabV3_ResNet101(num_classes, args, layers_to_hook=args.teacher_layers).to(args.device)
    t_model_3 = Create_DeepLabV3_ResNet101(num_classes, args, layers_to_hook=args.teacher_layers).to(args.device)

    # load pretrained teacher
    teacher_dict_1 = torch.load(args.teacher_path, map_location=args.device)["state_dict"]
    t_model_1.load_state_dict(teacher_dict_1)
    teacher_dict_2 = torch.load(args.teacher_path, map_location=args.device)["state_dict"]
    t_model_2.load_state_dict(teacher_dict_2)
    teacher_dict_2 = torch.load(args.teacher_path, map_location=args.device)["state_dict"]
    t_model_3.load_state_dict(teacher_dict_2)

    print(f"Creating student model: {args.model}...")
    if args.model == "enet":
        s_model = Create_ENet(num_classes, layers_to_hook=args.student_layers).to(args.device)
    elif args.model == "deeplabv3_mobilenetv3":
        s_model = Create_DeepLabV3_MobileNetV3(num_classes, args, layers_to_hook=args.student_layers).to(args.device)
    else:
        raise TypeError(f'Invalid model name. {args.model}')

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    criterion_list = nn.ModuleList([])
    module_list.append(s_model)
    trainable_list.append(s_model)
    criterion_kd = CriterionKD(args.kd_T)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_kd)

    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)

    module_list.append(t_model_1)
    module_list.append(t_model_2)
    module_list.append(t_model_3)
    module_list.to(args.device)
    criterion_list.to(args.device)
    
    lambda_lr = lambda iter: (1 - float(iter) / args.epochs) ** args.lr_decay
    lr_updater = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric_iou = metrics.IoU(num_classes, ignore_index=ignore_index)
    metric_pa = metrics.PixelAccuracy(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        raise ValueError("Resume mechanism is under construction")
        model, optimizer, start_epoch, best_miou, best_mpa = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        best_mpa = 0
        best_epoch = 0

    # Start Training
    print()
    distill = loop_ensemble.Distill_Ensemble(train_loader, module_list, criterion_list, optimizer, metric_iou, metric_pa, args)
    val = loops.Test(s_model, val_loader, criterion_list[0], metric_iou, metric_pa, args.device)
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {0:d}".format(epoch + 1))

        # train
        epoch_loss, cls_loss, kd_loss, (train_iou, train_miou), (train_pa, train_mpa), train_time = distill.run_epoch(epoch=epoch+1, iteration_loss=args.print_step)
        lr_updater.step()
        last_lr = lr_updater.get_last_lr()
        print("Result train: {0:d} => Avg. loss: {1:.4f} | CLS loss: {2:.4f} | KD loss: {3:.4f} | mIoU: {4:.4f} | mPA: {5:.4f} | lr: {6:.4f} | time elapsed: {7:.3f} seconds"\
              .format(epoch + 1, epoch_loss, cls_loss, kd_loss, train_miou, train_mpa, last_lr[0], train_time))

        # test
        test_loss, (test_iou, test_miou), (test_pa, test_mpa), test_time = val.run_epoch(args.print_step)
        print("Result Val: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f} | time elapsed: {4:.3f} seconds"\
              .format(epoch + 1, test_loss, test_miou, test_mpa, test_time))

        # Print per class IoU on last epoch or if best iou
        if test_miou > best_miou:
            for key, class_iou, class_pa in zip(class_encoding.keys(), test_iou, test_pa):
                print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

            # Save the model if it's the best thus far
            print("\nBest model based on mIoU thus far. Saving...\n")
            best_miou = test_miou
            best_mpa = test_mpa
            best_epoch = epoch
            utils.save_checkpoint(s_model, optimizer, epoch + 1, best_miou, best_mpa, args)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            images, _ = next(iter(val_loader))
            predict(s_model, images[:1], class_encoding, epoch)

    return s_model, best_epoch, best_miou


def predict(model, images, class_encoding, epoch):
    images = images.to(args.device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions, _ = model(images)
        if type(predictions) == OrderedDict:
            predictions = predictions["out"]

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    # utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':
    # elapsed_start_time = timer()
    # print(f"Teacher Result:")
    # inference(mode=args.mode)
    # print()

    # # seed everything
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # # Fail fast if the dataset directory doesn't exist
    # assert os.path.isdir(
    #     args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
    #         args.dataset_dir)

    # # Fail fast if the saving directory doesn't exist
    # assert os.path.isdir(
    #     args.save_dir), "The directory \"{0}\" doesn't exist.".format(
    #         args.save_dir)

    # # Import the requested dataset
    # if args.dataset.lower() == 'camvid':
    #     from data.camvid import CamVid as dataset
    # elif args.dataset.lower() == 'cityscapes':
    #     raise NameError('Dataset too big')
    #     # from data import Cityscapes as dataset
    # else:
    #     # Should never happen...but just in case it does
    #     raise RuntimeError("\"{0}\" is not a supported dataset.".format(
    #         args.dataset))

    # loaders, w_class, class_encoding = data_utils.load_dataset(dataset, args)
    # train_loader, val_loader, test_loader = loaders

    # model, epoch, miou = train_ensemble_kd(train_loader, val_loader, w_class, class_encoding, args)

    # elapsed_end_time = timer()
    # print(f"Elapsed time: {elapsed_end_time-elapsed_start_time:.3f} seconds")
    print("Kode ini belum selesai")