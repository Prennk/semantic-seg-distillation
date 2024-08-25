import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transforms import PILToLongTensor, LongTensorToRGBPIL
from PIL import Image
import numpy as np

from utils.utils import batch_transform, imshow_batch
from data.utils import enet_weighing, median_freq_balancing

def remove_unlabeled(label_tensor, unlabeled_value):
    """Remove the 'unlabeled' class from the label tensor."""
    # Set pixels with the 'unlabeled' value to a value outside valid range (e.g., -1)
    mask = label_tensor == unlabeled_value
    label_tensor[mask] = -2  # Assuming -1 is outside the valid range of class indices
    return label_tensor

def merge_road_marking(label_tensor, road_marking_value, road_value):
    """Merge 'road_marking' class with 'road' class in the label tensor."""
    mask = label_tensor == road_marking_value
    label_tensor[mask] = road_value  # Replace 'road_marking' class with 'road' class value
    return label_tensor

def load_dataset(dataset, args):
    print("Loading dataset...")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    # Tentukan nilai untuk 'unlabeled' jika ada di class_encoding
    unlabeled_value = None
    road_marking_value = None
    road_value = None

    # Load dataset sementara untuk mendapatkan encoding
    temp_dataset = dataset(args.dataset_dir)
    class_encoding = temp_dataset.color_encoding

    # Menghapus dan menggabungkan kelas sesuai dengan dataset
    if args.dataset.lower() == 'camvid':
        if 'road_marking' in class_encoding and 'road' in class_encoding:
            road_marking_value = list(class_encoding).index('road_marking')
            road_value = list(class_encoding).index('road')
            del class_encoding['road_marking']
            print(f"[Warning] Deleting 'road_marking' class because it is combined with 'road' class")

    # Jika 'unlabeled' ada di class encoding, tentukan nilai tensor-nya
    if 'unlabeled' in class_encoding:
        unlabeled_value = list(class_encoding).index('unlabeled')
        del class_encoding['unlabeled']
        print(f"[Info] 'unlabeled' class has been removed from class encoding.")

    # Update transformasi label untuk menghapus kelas 'unlabeled' dan menggabungkan kelas 'road_marking' dengan 'road'
    def label_processing_pipeline(label):
        if unlabeled_value is not None:
            label = remove_unlabeled(label, unlabeled_value)
        if road_marking_value is not None and road_value is not None:
            label = merge_road_marking(label, road_marking_value, road_value)
        return label

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        PILToLongTensor(),
        transforms.Lambda(label_processing_pipeline)
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

    # Get number of classes to predict
    num_classes = len(class_encoding)

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
            class_weights = enet_weighing(train_loader, num_classes, ignore_index=-1)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(args.device)
        print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding

# Update fungsi enet_weighing di utils/data/utils.py
def enet_weighing(dataloader, num_classes, ignore_index=None):
    """Computes the class weights using ENet's weighing method."""
    class_count = 0
    total = 0

    for _, label in dataloader:
        # Flatten label tensor and filter out 'ignore_index' if specified
        flat_label = label.view(-1)
        if ignore_index is not None:
            flat_label = flat_label[flat_label != ignore_index]

        # Compute class frequencies
        class_count += np.bincount(flat_label.cpu().numpy(), minlength=num_classes)
        total += flat_label.size(0)

    # Compute class weights
    class_weights = 1 / (class_count / total)
    return class_weights
