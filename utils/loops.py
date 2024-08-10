import torch
from tqdm import tqdm
from collections import OrderedDict
from timeit import default_timer as timer

class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric_iou, metric_pa, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric_iou = metric_iou
        self.metric_pa = metric_pa
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        start_time = timer()
        self.model.train()
        epoch_loss = 0.0
        self.metric_iou.reset()
        self.metric_pa.reset()
        for step, batch_data in enumerate(tqdm(self.data_loader, desc="Training")):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)

            if type(outputs) == OrderedDict:
                outputs = outputs['out']

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric_iou.add(outputs.detach(), labels.detach())
            self.metric_pa.add(outputs.detach(), labels.detach())

            end_time = timer()
            total_time = end_time - start_time

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric_iou.value(), self.metric_pa.value(), total_time
    

class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric_iou, metric_pa, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric_iou = metric_iou
        self.metric_pa = metric_pa
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        start_time = timer()
        self.model.eval()
        epoch_loss = 0.0
        self.metric_iou.reset()
        self.metric_pa.reset()
        for step, batch_data in enumerate(tqdm(self.data_loader, desc="Testing")):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)
                
                if type(outputs) == OrderedDict:
                    outputs = outputs['out']

                # Loss computation
                loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric_iou.add(outputs.detach(), labels.detach())
            self.metric_pa.add(outputs.detach(), labels.detach())

            end_time = timer()
            total_time = end_time - start_time

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric_iou.value(), self.metric_pa.value(), total_time