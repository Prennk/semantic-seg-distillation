import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transforms import PILToLongTensor, LongTensorToRGBPIL
from PIL import Image

from utils import batch_transform, imshow_batch
from data.utils import enet_weighing, median_freq_balancing

def load_dataset(dataset, args):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
        transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = batch_transform(labels, label_to_rgb)
        imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.model == 'ENet':
        if args.weighing.lower() == 'enet':
            class_weights = enet_weighing(train_loader, num_classes)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)
        else:
            class_weights = None
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(args.device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding
    