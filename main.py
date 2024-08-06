import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import yaml
from argparse import ArgumentParser
import os

from utils import utils, loops, metrics, transforms as ext_transforms, data_utils
from model.enet import Create_ENet
from model.deeplabv3 import Create_DeepLabV3

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

args = utils.merge_args_with_config(args, config)

def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    model = Create_ENet(num_classes).to(args.device)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric_iou = metrics.IoU(num_classes, ignore_index=ignore_index)
    metric_pa = metrics.PixelAccuracy(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou, best_mpa = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        best_mpa = 0

    # Start Training
    print()
    train = loops.Train(model, train_loader, optimizer, criterion, metric_iou, metric_pa, args.device)
    val = loops.Test(model, val_loader, criterion, metric_iou, metric_pa, args.device)
    for epoch in range(start_epoch, args.epochs):
        print("[TRAINING]Epoch: {0:d}".format(epoch + 1))

        epoch_loss, (iou, miou), (pa, mpa) = train.run_epoch(args.print_step)
        lr_updater.step()

        print("[RESULT]Epoch: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f}".format(epoch + 1, epoch_loss, miou, mpa))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print("[VALIDATION]Epoch: {0:d}".format(epoch + 1))

            loss, (iou, miou), (pa, mpa) = val.run_epoch(args.print_step)

            print("[RESULT]Epoch: {0:d} => Avg. loss: {1:.4f} | mIoU: {2:.4f} | mPA: {3:.4f}".format(epoch + 1, loss, miou, mpa))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
                    print("{0} => IoU: {1:.4f} | PA: {2:.4f}".format(key, class_iou, class_pa))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model based on mIoU thus far. Saving...\n")
                best_miou = miou
                best_mpa = mpa
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, best_mpa, args)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

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

    print("[TEST]Running test dataset")

    loss, (iou, miou), (pa, mpa) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))
    class_pa = dict(zip(class_encoding.keys(), pa))

    print("[RESULT]Avg. loss: {0:.4f} | mIoU: {1:.4f} | mPA: {2:.4f}".format(loss, miou, mpa))

    # Print per class IoU
    for key, class_iou, class_pa in zip(class_encoding.keys(), iou, pa):
        print("{:<15} => IoU: {:>10.4f} | PA: {:>10.4f}".format(key, class_iou, class_pa))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(args.device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)

# Run only if this module is being run directly
if __name__ == '__main__':

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
        model = train(train_loader, val_loader, w_class, class_encoding)

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

        if args.mode.lower() == 'test':
            # print(model)
            print(f"[Info] Model loaded succesfully")

        test(model, test_loader, w_class, class_encoding)

# to do => PA: nan, load weight