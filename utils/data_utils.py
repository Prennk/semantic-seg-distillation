import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transforms import PILToLongTensor, LongTensorToRGBPIL
from PIL import Image

from utils.utils import batch_transform, imshow_batch
from data.utils import enet_weighing, median_freq_balancing

def remove_unlabeled(label_tensor, unlabeled_value, num_classes):
    """Remove the 'unlabeled' class from the label tensor by setting it to a new default value."""
    new_value = num_classes  # New value should be out of the valid class range
    label_tensor[label_tensor == unlabeled_value] = new_value
    return label_tensor

def load_dataset(dataset, args):
    print("Loading dataset...")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    temp_image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    temp_label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        PILToLongTensor()])

    temp_train_set = dataset(
        args.dataset_dir,
        transform=temp_image_transform,
        label_transform=temp_label_transform)

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = temp_train_set.color_encoding
    unlabeled_index = None

    if args.ignore_unlabeled and 'unlabeled' in class_encoding:
        unlabeled_index = list(class_encoding.keys()).index('unlabeled')
        del class_encoding['unlabeled']
        print(f"[Info] 'unlabeled' class has been removed from class encoding.")

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    num_classes = len(class_encoding)
    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        PILToLongTensor(),
        transforms.Lambda(lambda label: remove_unlabeled(label, unlabeled_index, num_classes) if unlabeled_index is not None else label)
    ])

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

    # Remove the road_marking class from the CamVid dataset as it's merged with the road class
    if args.dataset.lower() == 'camvid':
        if 'road_marking' in class_encoding:
            del class_encoding['road_marking']
            print(f"[Warning] Deleting 'road_marking' class because it is combined with 'road' class")

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))
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
    print("Weighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = None
    if args.weighing:
        if args.weighing.lower() == 'enet':
            class_weights = enet_weighing(train_loader, num_classes)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(args.device)
        if args.ignore_unlabeled and unlabeled_index is not None:
            # Remove the weight for 'unlabeled' class using the saved index
            class_weights = torch.cat((class_weights[:unlabeled_index], class_weights[unlabeled_index+1:]), dim=0)
            print(f"[Warning] Removed 'unlabeled' class weight from class weights.")

        print(f"Num of class weights: {len(class_weights)}")
        print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding
