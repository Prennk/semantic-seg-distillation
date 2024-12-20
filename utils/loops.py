import torch
import torch.nn as nn
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
        total_inference_time = 0.0
        total_images = 0
        for step, batch_data in enumerate(tqdm(self.data_loader, desc="Testing")):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            total_images += inputs.size(0)

            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.synchronize()
                # Forward propagation
                inference_start = timer()
                outputs, _ = self.model(inputs)
                inference_end = timer()
                
                if isinstance(outputs, OrderedDict):
                    outputs = outputs['out']
                elif isinstance(outputs, list):
                    outputs = outputs[0]

                total_inference_time += inference_end - inference_start

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
                print("[Step: %d] Iteration loss: %.2f" % (step, loss.item()))

        inference_speed_per_image = (total_inference_time / total_images) * 1000
        print(f"Inference speed: {inference_speed_per_image:.4f} ms per image")

        return epoch_loss / len(self.data_loader), self.metric_iou.value(), self.metric_pa.value(), total_time

    def export_model(self, export_path="model_export.onnx"):
        """
        Exports the model to ONNX format.

        Args:
            input_sample (torch.Tensor): A sample input tensor to define the model's input shape.
            export_path (str, optional): Path to save the exported ONNX model. Defaults to "model_export.onnx".
        """
        self.model.eval()

        # Define dummy inputs for tracing (dynamic batch size)
        dummy_input = torch.randn(1, 3, 360, 480).to(self.device)

        # Export the traced model to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            opset_version=20,  # Adjust based on your target runtime
            input_names=["input"],  # Name your input
            output_names=["output"],  # Name your output
        )

        print(f"Model has been successfully exported to {export_path}.")


    def evaluate_exported_model(self, export_path="model_export.onnx"):
        """
        Evaluates the exported ONNX model.

        Args:
            export_path (str, optional): Path to the exported ONNX model. Defaults to "model_export.onnx".

        Returns:
            float: Inference speed per image in milliseconds.
        """

        # Load the exported model with an ONNX runtime
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(export_path)
        input_name = sess.get_inputs()[0].name  # Get input name from session

        total_inference_time = 0.0
        total_images = 0

        for step, batch_data in enumerate(tqdm(self.data_loader, desc="Testing Exported Model")):
            inputs = batch_data[0].cpu()
            total_images += inputs.size(0)

            # Prepare data for ONNX runtime (might need conversion based on runtime)
            ort_inputs = {input_name: inputs.numpy()}

            # Run inference
            inference_start = timer()
            outputs = sess.run(None, ort_inputs)[0]  # Assuming single output
            inference_end = timer()

            total_inference_time += inference_end - inference_start

        inference_speed_per_image = (total_inference_time / total_images) * 1000
        print(f"Inference speed for exported model: {inference_speed_per_image:.2f} ms per image")

        return inference_speed_per_image


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

    def __init__(self, data_loader, module_list, criterion_list, optim, metric_iou, metric_pa, args):
        self.data_loader = data_loader
        self.module_list = module_list
        self.criterion_list = criterion_list
        self.optim = optim
        self.metric_iou = metric_iou
        self.metric_pa = metric_pa
        self.args = args

    def run_epoch(self, epoch, iteration_loss=False):
        """Runs an epoch of training."""
        start_time = timer()

        for module in self.module_list:
            module.train()

        self.module_list[-1].eval()

        criterion_cls = self.criterion_list[0]
        criterion_kd = self.criterion_list[1]

        s_model = self.module_list[0]
        t_model = self.module_list[-1]

        epoch_loss = 0.0
        cls_loss = 0.0
        kd_loss = 0.0
        self.metric_iou.reset()
        self.metric_pa.reset()

        for step, batch_data in enumerate(tqdm(self.data_loader, desc=f"Training ({self.args.distillation})")):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.args.device)
            labels = batch_data[1].to(self.args.device)

            # Forward propagation for teacher
            with torch.no_grad():
                t_outputs, t_intermediate_features = t_model(inputs)
                if isinstance(t_outputs, OrderedDict):
                    t_aux_outputs = t_outputs["aux"]
                    t_outputs = t_outputs['out']

            # Forward propagation for student
            s_outputs, s_intermediate_features = s_model(inputs)
            if isinstance(s_outputs, OrderedDict):
                s_aux_outputs = s_outputs["aux"]
                s_outputs = s_outputs['out']

            # Loss computation
            if isinstance(s_outputs, OrderedDict):
                loss_aux = criterion_cls(s_aux_outputs, labels)
                loss_cls = criterion_cls(s_outputs, labels)
                loss_cls_total = (0.4 * loss_aux) + loss_cls
            else:
                loss_cls = criterion_cls(s_outputs, labels)
                loss_cls_total = criterion_cls(s_outputs, labels)

            if self.args.distillation == "kd":
                if isinstance(s_outputs, OrderedDict):
                    loss_div_aux = criterion_kd(s_aux_outputs, t_aux_outputs)
                    loss_div = criterion_kd(s_outputs, t_outputs)
                    loss_kd = (0.4 * loss_div_aux) + loss_div
                else:
                    loss_kd = criterion_kd(s_outputs, t_outputs)

                loss = (self.args.gamma * loss_cls_total) + (self.args.alpha * loss_kd)
            elif self.args.distillation == "vid":
                feat_t = [v.detach() for k, v in t_intermediate_features.items()] + [t_outputs.detach()]
                feat_s = [v for k, v in s_intermediate_features.items()] + [s_outputs]

                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(feat_s, feat_t, criterion_kd)]
                loss_kd = sum(loss_group)
                
                loss = loss_cls_total + loss_kd
            elif self.args.distillation == "fsp":
                loss_group = criterion_kd(s_intermediate_features, t_intermediate_features)
                loss_kd = sum(loss_group)

                loss = loss_cls_total = loss_kd
            elif self.args.distillation == "dtkd":
                if isinstance(s_outputs, OrderedDict):
                    loss_div_aux = criterion_kd(s_aux_outputs, t_aux_outputs)
                    loss_div = criterion_kd(s_outputs, t_outputs)
                    loss_kd = (0.4 * loss_div_aux) + loss_div
                else:
                    loss_kd = criterion_kd(s_outputs, t_outputs)

                loss = (self.args.gamma * loss_cls_total) + (self.args.alpha * loss_kd)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            cls_loss += loss_cls.item()
            kd_loss += loss_kd.item()

            # Keep track of the evaluation metric
            self.metric_iou.add(s_outputs.detach(), labels.detach())
            self.metric_pa.add(s_outputs.detach(), labels.detach()) 

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        end_time = timer()
        total_time = end_time - start_time

        return epoch_loss / len(self.data_loader), cls_loss / len(self.data_loader), kd_loss / len(self.data_loader), self.metric_iou.value(), self.metric_pa.value(), total_time
    
