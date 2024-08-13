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
import wandb

from utils import utils, loops, metrics, transforms as ext_transforms, data_utils
from model.enet import Create_ENet
from model.deeplabv3 import Create_DeepLabV3
from distiller.vid import VIDLoss

# init wandb
wandb.init(project='SemSeg-Distill')

# get config from config.yaml
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
args = utils.merge_args_with_config(args, config)

# save config to wandb
wandb.config.update(args)

# print config from config.yaml
print("-" * 70)
print("| Config" + " " * 61 + "|")
print("-" * 70)
for key, value in vars(args).items():
    print(f"| {key}: {value}" + " " * (67 - len(f"{key}: {value}")) + "|")
print("-" * 70)
print()

# seed everything
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)


def train(train_loader, val_loader, class_weights, class_encoding, args):
    print(f"\nCreating model: {args.model}")

    num_classes = len(class_encoding)

    # Intialize model
    if args.model == 'enet':
        model = Create_ENet(num_classes).to(args.device)
    elif args.model == 'deeplabv3':
        model = Create_DeepLabV3(num_classes, args).to(args.device)
    else:
        raise TypeError('Invalid model name. Available models are enet and deeplabv3')
    
    # send model to wandb
    # wandb.watch(model, log="all")

    # print model summary
    model_summary = summary(model=model,
                            input_data=torch.randn(1, 3, args.width, args.height).to(args.device),
                            col_names=["trainable"],
                            row_settings=["var_names"])
    print(model_summary)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    # lr_updater = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
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
    train = loops.Train(model, train_loader, optimizer, criterion, metric_iou, metric_pa, args.device)
    val = loops.Test(model, val_loader, criterion, metric_iou, metric_pa, args.device)
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {0:d}".format(epoch + 1))

        epoch_loss, (iou, miou), (pa, mpa), train_time = train.run_epoch(args.print_step)
        lr_updater.step()
        last_lr = lr_updater.get_last_lr()

        print("Result train: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f} | lr: {4} | time elapsed: {5:.3f} seconds"\
              .format(epoch + 1, epoch_loss, miou, mpa, last_lr[0], train_time))

        # send train metric results to wandb
        wandb.log({
            "train_loss": epoch_loss,
            "train_miou": miou,
            "train_mpa": mpa,
            }, step=epoch + 1)

        loss, (iou, miou), (pa, mpa), test_time = val.run_epoch(args.print_step)
        print("Result Val: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f} | time elapsed: {4:.3f} seconds"\
              .format(epoch + 1, loss, miou, mpa, test_time))

        # send val metric results to wandb
        wandb.log({
            "val_loss": loss,
            "val_miou": miou,
            "val_mpa": mpa
            }, step=epoch + 1)

        # Print per class IoU on last epoch or if best iou
        if miou > best_miou:
            for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
                print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model based on mIoU thus far. Saving...\n")
                best_miou = miou
                best_mpa = mpa
                best_epoch = epoch
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, best_mpa, args)

        if epoch + 1 % 10 == 0 or epoch + 1 == args.epochs:
            # predict the segmentation map and send it to wandb
            images, _ = next(iter(val_loader))
            predict(model, images[:1], class_encoding, epoch)

    return model, best_epoch, best_miou


def test(model, test_loader, class_weights, class_encoding):
    print("\nStart testing...")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric_iou = metrics.IoU(num_classes, ignore_index=ignore_index)
    metric_pa = metrics.PixelAccuracy(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = loops.Test(model, test_loader, criterion, metric_iou, metric_pa, args.device)

    print("Running test dataset...")

    loss, (iou, miou), (pa, mpa) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))
    class_pa = dict(zip(class_encoding.keys(), pa))

    print("Result => Avg. loss: {0:.4f} | mIoU: {1:.4f} | mPA: {2:.4f}".format(loss, miou, mpa))

    # send val metric results to wandb
    wandb.log({
        "test_loss": loss,
        "test_miou": miou,
        "test_mpa": mpa
        })

    # Print per class IoU
    for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
        print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("[Warning] imshow_batch is deprecated")
        # print("A batch of predictions from the test set...")
        # images, _ = next(iter(test_loader))
        # predict(model, images, class_encoding)


def predict(model, images, class_encoding, epoch):
    images = images.to(args.device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)
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

    # send visualization to wandb
    wandb.log({
        "segmentation_map": [wandb.Image(image, caption="Segmentation Map") for image in color_predictions]
    }, step=epoch + 1)


def distill(train_loader, val_loader, class_weights, class_encoding, args):
    num_classes = len(class_encoding)

    # create teacher
    print(f"\nLoading teacher model: deeplabv3 from {args.teacher_path}...")
    t_model = Create_DeepLabV3(num_classes, args, layers_to_hook=args.teacher_layers).to(args.device)

    # load pretrained teacher
    teacher_dict = torch.load(args.teacher_path, map_location=args.device)["state_dict"]
    t_model.load_state_dict(teacher_dict)

    print(f"Creating student model: enet...")
    s_model = Create_ENet(num_classes, layers_to_hook=args.student_layers).to(args.device)
    
    # send model to wandb
    # wandb.watch(model, log="all")

    # print layer to distill
    print(f"\nTeacher layer to distill: \n{args.teacher_layers}")
    print(f"Student layer to distill: \n{args.student_layers}")

    if len(args.teacher_layers) != len(args.student_layers):
        raise ValueError("Number of layers to distill in teacher and student models do not match.")

    # get layers size for VID
    t_shapes = [t_model.get_feature_map(layer).shape for layer in args.teacher_layers]
    s_shapes = [s_model.get_feature_map(layer).shape for layer in args.student_layers]

    print(f"Teacher layer shapes: {[shape for shape in t_shapes]}")
    print(f"Student layer shapes: {[shape for shape in s_shapes]}")

    vid_criterions = nn.ModuleList([VIDLoss(s_shape, t_shape, t_shape) for s_shape, t_shape in zip(s_shapes, t_shapes)])
    
    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.SGD(
        s_model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)
    
    # Learning rate decay scheduler
    # lr_updater = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
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
    distill = loops.Distill(
        t_model, s_model, train_loader, optimizer, criterion, 
        vid_criterions, metric_iou, metric_pa, args.device)
    val = loops.Test(s_model, val_loader, criterion, metric_iou, metric_pa, args.device)
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {0:d}".format(epoch + 1))

        epoch_loss, (iou, miou), (pa, mpa), train_time = distill.run_epoch(args.print_step)
        lr_updater.step()
        last_lr = lr_updater.get_last_lr()

        print("Result train: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f} | lr: {4} | time elapsed: {5:.3f} seconds"\
              .format(epoch + 1, epoch_loss, miou, mpa, last_lr[0], train_time))

        # send train metric results to wandb
        wandb.log({
            "train_loss": epoch_loss,
            "train_miou": miou,
            "train_mpa": mpa,
            }, step=epoch + 1)

        loss, (iou, miou), (pa, mpa), test_time = val.run_epoch(args.print_step)
        print("Result Val: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f} | time elapsed: {4:.3f} seconds"\
              .format(epoch + 1, loss, miou, mpa, test_time))

        # send val metric results to wandb
        wandb.log({
            "val_loss": loss,
            "val_miou": miou,
            "val_mpa": mpa
            }, step=epoch + 1)

        # Print per class IoU on last epoch or if best iou
        if miou > best_miou:
            for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
                print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model based on mIoU thus far. Saving...\n")
                best_miou = miou
                best_mpa = mpa
                best_epoch = epoch
                utils.save_checkpoint(s_model, optimizer, epoch + 1, best_miou, best_mpa, args)

        if epoch + 1 % 10 == 0 or epoch + 1 == args.epochs:
            # predict the segmentation map and send it to wandb
            images, _ = next(iter(val_loader))
            predict(s_model, images[:1], class_encoding, epoch)

    return s_model, best_epoch, best_miou


# Run only if this module is being run directly
if __name__ == '__main__':
    elapsed_start_time = timer()

    # seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data.camvid import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        raise NameError('Dataset too big')
        # from data import Cityscapes as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    loaders, w_class, class_encoding = data_utils.load_dataset(dataset, args)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model, epoch, miou = train(train_loader, val_loader, w_class, class_encoding, args)
        print(f"Best mIoU: {miou} in epoch {epoch}")

    if args.mode.lower() == "distill":
        model, epoch, miou = distill(train_loader, val_loader, w_class, class_encoding, args)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            model = Create_ENet(num_classes).to(args.device)


        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        test(model, test_loader, w_class, class_encoding)

    elapsed_end_time = timer()
    print(f"Elapsed time: {elapsed_end_time-elapsed_start_time:.3f} seconds")