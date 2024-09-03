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
            outputs, _ = self.model(inputs)

            if isinstance(outputs, OrderedDict):
                aux_outputs = outputs['aux']
                classifier_outputs = outputs['out']
                # Loss computation
                aux_loss = self.criterion(aux_outputs, labels)
                classifier_loss = self.criterion(classifier_outputs, labels)
                total_loss = (0.4 * aux_loss) + classifier_loss

                # Keep track of the evaluation metric
                self.metric_iou.add(classifier_outputs.detach(), labels.detach())
                self.metric_pa.add(classifier_outputs.detach(), labels.detach())
            else:
                loss = self.criterion(outputs, labels)
                total_loss = loss

                # Keep track of the evaluation metric
                self.metric_iou.add(outputs.detach(), labels.detach())
                self.metric_pa.add(outputs.detach(), labels.detach())

            # Backpropagation
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += total_loss.item()

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
                result = self.model(inputs)
                if isinstance(result, tuple) and len(result) == 3:
                    outputs, _, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    outputs, _ = result
                
                if isinstance(outputs, OrderedDict):
                    outputs = outputs['out']
                elif isinstance(outputs, list):
                    outputs = outputs[0]

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


class Distill:
    """Performs the distillation of from teacher to student model given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - t_model (``nn.Module``): the teacher model instance.
    - s_model (``nn.Module``): the student model instance.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - criterion (``Optimizer``): The distillation loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, t_model, s_model, data_loader, optim, criterion, distill_criterion, metric_iou, metric_pa, device):
        self.t_model = t_model
        self.s_model = s_model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.distill_criterion = distill_criterion
        self.metric_iou = metric_iou
        self.metric_pa = metric_pa
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training."""
        start_time = timer()

        self.t_model.eval()
        self.s_model.train()

        epoch_loss = 0.0
        self.metric_iou.reset()
        self.metric_pa.reset()

        for step, batch_data in enumerate(tqdm(self.data_loader, desc="Training (distillation)")):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation for teacher
            with torch.no_grad():
                t_outputs, t_intermediate_features = self.t_model(inputs)
                if isinstance(t_outputs, OrderedDict):
                    t_outputs = t_outputs['out']

            # Forward propagation for student
            s_outputs, s_inter, s_intermediate_features = self.s_model(inputs)
            if isinstance(s_outputs, OrderedDict):
                s_outputs = s_outputs['out']

            # Loss computation
            loss = self.criterion(s_outputs, labels)

            # Reset distill_loss for each batch
            distill_loss = 0.0

            # Distill loss
            self.distill_criterion.to(self.device)
            for idx, (t_layer_name, s_layer_name) in enumerate(zip(self.t_model.layers_to_hook, self.s_model.layers_to_hook)):
                t_features = t_intermediate_features[t_layer_name]
                # s_features = s_intermediate_features[s_layer_name]
                s_features = s_inter[idx]
                distill_loss += self.distill_criterion[idx](s_features, t_features)
                
            # Total loss
            total_loss = loss + distill_loss

            # Backpropagation
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += total_loss.item()

            # Keep track of the evaluation metric
            self.metric_iou.add(s_outputs.detach(), labels.detach())
            self.metric_pa.add(s_outputs.detach(), labels.detach()) 

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, total_loss.item()))

        end_time = timer()
        total_time = end_time - start_time

        return epoch_loss / len(self.data_loader), self.metric_iou.value(), self.metric_pa.value(), total_time