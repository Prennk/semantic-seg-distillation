import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transforms import PILToLongTensor, LongTensorToRGBPIL
from PIL import Image

from utils.utils import batch_transform, imshow_batch
from data.utils import enet_weighing, median_freq_balancing

def remove_unlabeled(label_tensor, unlabeled_value):
    """Remove the 'unlabeled' class from the label tensor."""
    # Set pixels with the 'unlabeled' value to a value outside valid range (e.g., -2)
    mask = label_tensor == unlabeled_value
    label_tensor[mask] = -2 
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

    # Load dataset untuk mendapatkan encoding
    temp_dataset = dataset(args.dataset_dir)
    class_encoding = temp_dataset.color_encoding

    # Jika 'unlabeled' ada di class encoding, tentukan nilai tensor-nya
    if 'unlabeled' in class_encoding:
        unlabeled_value = list(class_encoding).index('unlabeled')
        del class_encoding['unlabeled']
        print(f"[Info] 'unlabeled' class has been removed from class encoding.")

    # Update transformasi label untuk menghapus kelas 'unlabeled'
    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        PILToLongTensor(),
        transforms.Lambda(lambda label: remove_unlabeled(label, unlabeled_value) if unlabeled_value is not None else label)
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
            class_weights = enet_weighing(train_loader, num_classes)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(args.device)
        print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding
