import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from timeit import default_timer as timer

class Distill_Ensemble:
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
        t_model_1 = self.module_list[1]
        t_model_2 = self.module_list[2]
        t_model_3 = self.module_list[3]

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
                t_outputs_1, t_intermediate_features_1 = t_model_1(inputs)
                t_outputs_2, t_intermediate_features_2 = t_model_2(inputs)
                t_outputs_3, t_intermediate_features_3 = t_model_3(inputs)
                if isinstance(t_outputs_1, OrderedDict):
                    t_aux_outputs_1 = t_outputs_1["aux"]
                    t_outputs_1 = t_outputs_1['out']
                if isinstance(t_outputs_2, OrderedDict):
                    t_aux_outputs_2 = t_outputs_2["aux"]
                    t_outputs_2 = t_outputs_2['out']
                if isinstance(t_outputs_3, OrderedDict):
                    t_aux_outputs_3 = t_outputs_3["aux"]
                    t_outputs_3 = t_outputs_3['out']

                t_aux_outputs_contenated = torch.stack([t_aux_outputs_1, t_aux_outputs_2, t_aux_outputs_3], dim=0)
                t_aux_outputs_voted = torch.mode(t_aux_outputs_contenated, dim=0)
                t_outputs_concatenated = torch.stack([t_outputs_1, t_outputs_2, t_outputs_3], dim=0)
                t_outputs_voted, _ = torch.mode(t_outputs_concatenated, dim=0)

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

            if isinstance(s_outputs, OrderedDict):
                loss_div_aux = criterion_kd(s_aux_outputs, t_aux_outputs_voted)
                loss_div = criterion_kd(s_outputs, t_outputs_voted)
                loss_kd = (0.4 * loss_div_aux) + loss_div
            else:
                loss_kd = criterion_kd(s_outputs, t_outputs_voted)

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