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

# ===========================================================================
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )

        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def sobel_edge_detection(self, x):
        sobel_x = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).to(x.device)
        sobel_y = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).to(x.device)

        sobel_x = sobel_x.expand(x.size(1), 1, 3, 3)
        sobel_y = sobel_y.expand(x.size(1), 1, 3, 3)
        
        edge_x = F.conv2d(x, sobel_x, groups=x.size(1), padding=1)
        edge_y = F.conv2d(x, sobel_y, groups=x.size(1), padding=1)
        
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge_magnitude

    def forward(self, input, target):
        # pool for dimension match
        s_H, s_W = input.shape[2], input.shape[3]
        t_H, t_W = target.shape[2], target.shape[3]

        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_W))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_W))
        else:
            pass

        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        target_edge = self.sobel_edge_detection(target)
        input_edge = self.sobel_edge_detection(pred_mean)
        edge_threshold = 0.4
        edge_mask = (target_edge > edge_threshold).float()
        loss = 0.5*(
            (input_edge-target_edge)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(loss * edge_mask)

        return loss