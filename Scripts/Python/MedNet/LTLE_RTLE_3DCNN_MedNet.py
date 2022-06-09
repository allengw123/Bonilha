# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
 

trained_model = Net()
trained_model.load_state_dict(torch.load('mnist.pth'))
dummy_input = Variable(torch.randn(1, 1, 28, 28))
torch.onnx.export(trained_model, dummy_input, "mnist.onnx")
 
model = onnx.load('mnist.onnx')
tf_rep = prepare(model) 