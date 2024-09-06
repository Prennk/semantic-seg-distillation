import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt

def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, mpa, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - mpa (``float``): The mean pixel accuracy obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'mpa': mpa,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Loads the model and optimizer state from a specified file.

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The optimizer instance where only the model's
    optimizer state will be copied (excluding additional components like VIDLoss).
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, mPA, ``model``, and ``optimizer`` loaded from the
    checkpoint.
    """
    assert os.path.isdir(
        folder_dir), f"The directory \"{folder_dir}\" doesn't exist."

    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), f"The model file \"{filename}\" doesn't exist."
    
    # Load the stored model parameters to the model instance
    print(f'Loading model from {folder_dir}/{filename}...')
    checkpoint = torch.load(model_path)

    # Load the model state
    model.load_state_dict(checkpoint['state_dict'])

    # Load the optimizer state for the model only (filter parameters)
    optimizer_state_dict = checkpoint['optimizer']
    optimizer_param_groups = optimizer_state_dict['param_groups']

    # Filter only the parameters related to the model
    model_param_ids = {id(p) for p in model.parameters()}
    filtered_param_groups = []
    
    for param_group in optimizer_param_groups:
        filtered_params = [p for p in param_group['params'] if id(p) in model_param_ids]
        if filtered_params:
            param_group['params'] = filtered_params
            filtered_param_groups.append(param_group)

    optimizer_state_dict['param_groups'] = filtered_param_groups
    optimizer.load_state_dict(optimizer_state_dict)

    # Load other info like epoch, mIoU, and mPA
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    mpa = checkpoint.get('mpa', 0.0)

    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Epoch: {epoch}")
    print(f"mIOU: {miou}")
    print(f"mPA: {mpa}")
    print('Model and optimizer loaded successfully')

    return model, optimizer, epoch, miou, mpa

def merge_args_with_config(args, config):
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


