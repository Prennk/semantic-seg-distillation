import os
import numpy as np
from collections import OrderedDict
import torch
import torch.utils.data as data
from . import utils


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        # ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        # ('unlabeled', (0, 0, 0))
    ])

    color_to_label = {
        (128, 128, 128): 0,  # sky
        (128, 0, 0): 1,      # building
        (192, 192, 128): 2,  # pole
        # (255, 69, 0): 3,   # road_marking
        (128, 64, 128): 3,   # road
        (60, 40, 222): 4,    # pavement
        (128, 128, 0): 5,    # tree
        (192, 128, 128): 6,  # sign_symbol
        (64, 64, 128): 7,    # fence
        (64, 0, 128): 8,     # car
        (64, 64, 0): 9,      # pedestrian
        (0, 128, 192): 10,   # bicyclist
        # (0, 0, 0):11
    }


    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader,
                 color_to_label=color_to_label,
                 ignore_index=255):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.color_to_label = color_to_label
        self.ignore_index = ignore_index

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        label_np = np.array(label)

        # Mapping road_marking to road and ignoring unlabeled
        road_marking_color = (255, 69, 0)
        road_color = (128, 64, 128)
        road_label = self.color_to_label[road_color]

        mask = np.all(label_np == road_marking_color, axis=-1)
        label_np[mask] = road_label

        # Convert any pixel not in color_encoding to ignore_index
        valid_labels = list(self.color_to_label.values()) + [self.ignore_index]
        label_np = np.where(np.isin(label_np, valid_labels, invert=True), self.ignore_index, label_np)

        # Convert back to torch tensor
        label = torch.from_numpy(label_np).long()

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")