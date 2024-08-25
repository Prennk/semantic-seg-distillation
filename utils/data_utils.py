import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transforms import PILToLongTensor, LongTensorToRGBPIL
from PIL import Image

from utils.utils import batch_transform, imshow_batch
from data.utils import enet_weighing, median_freq_balancing

def load_dataset(dataset, args):
    print("Loading dataset...")
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
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

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

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        if 'road_marking' in class_encoding:
            del class_encoding['road_marking']
            print(f"[Warning] Deleting 'road_marking' class because it is combined with 'road' class")

    # Check if 'unlabeled' class needs to be ignored
    ignore_unlabeled = args.ignore_unlabeled
    if ignore_unlabeled and 'unlabeled' in class_encoding:
        ignore_index = list(class_encoding).index('unlabeled')
        # Remove 'unlabeled' from class encoding
        del class_encoding['unlabeled']
        print(f"[Warning] Deleting 'unlabeled' class from class encoding")

        # Remove 'unlabeled' from class weights
        class_weights = 0
        if args.weighing:
            if args.weighing.lower() == 'enet':
                class_weights = enet_weighing(train_loader, len(class_encoding))
            elif args.weighing.lower() == 'mfb':
                class_weights = median_freq_balancing(train_loader, len(class_encoding))
            else:
                class_weights = None
        else:
            class_weights = None

        if class_weights is not None:
            class_weights = torch.from_numpy(class_weights).float().to(args.device)
    else:
        # Get number of classes to predict
        num_classes = len(class_encoding)
        class_weights = 0
        if args.weighing:
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

    print("Number of classes to predict:", len(class_encoding))
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))

    # Remove 'unlabeled' class from labels if present
    if ignore_unlabeled and 'unlabeled' in class_encoding:
        labels = (labels != ignore_index).long() * labels
        labels = labels[labels != ignore_index]

    print("Image size:", images.size())
    print("Label size:", labels.size())
    print(f"Num of class weights: {len(class_weights)}")
    print("Class-color encoding:", class_encoding)

    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = batch_transform(labels, label_to_rgb)
        imshow_batch(images, color_labels)

    print("Weighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")

    if class_weights is not None:
        print(f"Num of class weights: {len(class_weights)}")
        print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding
