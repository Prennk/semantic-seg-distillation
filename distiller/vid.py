from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 args,
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

        if args.regressor_type == "default":
            print("VID regressor: default")
            self.regressor = nn.Sequential(
                conv1x1(num_input_channels, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_target_channels),
            )
        elif args.regressor_type == "medium":
            print("VID regressor: medium")
            self.regressor = nn.Sequential(
                conv1x1(num_input_channels, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),

                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),

                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),

                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),

                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),
                conv1x1(num_mid_channel, num_mid_channel),
                nn.ReLU(),

                conv1x1(num_mid_channel, num_target_channels),
            )
        else:
            raise ValueError("Please provide a valid regressor type")

        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

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
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss