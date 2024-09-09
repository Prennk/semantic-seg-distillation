# from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class VIDLoss(nn.Module):
#     """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
#     code from author: https://github.com/ssahn0215/variational-information-distillation"""
#     def __init__(self,
#                  num_input_channels,
#                  num_mid_channel,
#                  num_target_channels,
#                  init_pred_var=5.0,
#                  eps=1e-5):
#         super(VIDLoss, self).__init__()

#         def conv1x1(in_channels, out_channels, stride=1):
#             return nn.Conv2d(
#                 in_channels, out_channels,
#                 kernel_size=1, padding=0,
#                 bias=False, stride=stride)

#         self.regressor = nn.Sequential(
#             conv1x1(num_input_channels, num_mid_channel),
#             nn.ReLU(),
#             conv1x1(num_mid_channel, num_mid_channel),
#             nn.ReLU(),
#             conv1x1(num_mid_channel, num_target_channels),
#         )

#         self.log_scale = torch.nn.Parameter(
#             np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
#             )
#         self.eps = eps

#     def forward(self, input, target):
#         # pool for dimension match
#         s_H, s_W = input.shape[2], input.shape[3]
#         t_H, t_W = target.shape[2], target.shape[3]

#         if s_H > t_H:
#             input = F.adaptive_avg_pool2d(input, (t_H, t_W))
#         elif s_H < t_H:
#             target = F.adaptive_avg_pool2d(target, (s_H, s_W))
#         else:
#             pass

#         pred_mean = self.regressor(input)
#         pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
#         pred_var = pred_var.view(1, -1, 1, 1)
#         neg_log_prob = 0.5*(
#             (pred_mean-target)**2/pred_var+torch.log(pred_var)
#             )
#         loss = torch.mean(neg_log_prob)
#         return loss

# ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VIDLossWithMask(nn.Module):
    """Modified Variational Information Distillation with Masking for each class"""
    def __init__(self, num_input_channels, num_mid_channel, num_target_channels, num_classes, init_pred_var=5.0, eps=1e-5):
        super(VIDLossWithMask, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride)
        
        # Regressor untuk student
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
            nn.ReLU(),
            conv1x1(num_target_channels, num_classes),
        )

        # Regressor untuk teacher (untuk mereduksi channel)
        self.teacher_reduction = conv1x1(num_target_channels, num_classes)

        # Variance log scale untuk likelihood estimation
        self.log_scale = torch.nn.Parameter(np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_classes))
        self.eps = eps
        self.num_classes = num_classes

    def generate_class_mask(self, labels, height, width):
        """
        Generate class masks from ground truth labels.
        Args:
            labels: Ground truth segmentation map (B x H x W), where each pixel is a class index.
            height, width: Size of the feature maps that masks will be applied to.

        Returns:
            class_masks: A tensor of shape (B x num_classes x H x W) containing one-hot masks for each class.
        """
        labels_resized = F.interpolate(labels.unsqueeze(1).float(), size=(height, width), mode='nearest').squeeze(1).long()

        B = labels_resized.size(0)
        class_masks = torch.zeros((B, self.num_classes, height, width), device=labels_resized.device)

        for cls in range(self.num_classes):
            class_masks[:, cls, :, :] = (labels_resized == cls).float()

        return class_masks

    def forward(self, input, target, labels):
        # Pooling for dimension match if needed
        s_H, s_W = input.shape[2], input.shape[3]
        t_H, t_W = target.shape[2], target.shape[3]

        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_W))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_W))
        else:
            pass

        # Regress the student's feature map to match teacher's
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        # Regress teacher's feature map to reduce its channel to num_classes
        reduced_target = self.teacher_reduction(target)

        # Generate class masks from labels
        class_masks = self.generate_class_mask(labels, target.shape[2], target.shape[3])

        total_loss = 0.0

        # Loop over each class
        for cls in range(self.num_classes):
            # Apply the mask for the current class
            mask = class_masks[:, cls, :, :].unsqueeze(1)  # (B x 1 x H x W)
            mask = mask.expand_as(pred_mean)  # Broadcast mask to the same shape as feature maps

            # Apply the mask to the predicted mean and reduced target
            masked_pred_mean = pred_mean * mask
            masked_target = reduced_target * mask

            # Calculate negative log-likelihood for the current class
            neg_log_prob = 0.5 * ((masked_pred_mean - masked_target) ** 2 / pred_var + torch.log(pred_var))
            loss = torch.mean(neg_log_prob)  # Average over all spatial locations

            total_loss += loss

        return total_loss / self.num_classes
