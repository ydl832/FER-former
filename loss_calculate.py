# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 06:08:19 2022

@author: Administrator
"""

import torch.nn as nn
import torch
# import numpy as np
import torch.nn.functional as F


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
    
    def forward(self, sembed, logit, class_label, label, values):

        loss0 = self.criterion(sembed, label)
        loss1 = self.criterion(values, label)

        return loss0 + loss1, loss1



