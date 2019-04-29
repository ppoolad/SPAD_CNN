# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:46:04 2019

@author: ppool
"""

import torch
import torch.nn as nn
import numpy as np
import skimage.transform
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor


input_var = Variable(torch.randn(1, 1, 32, 16, 16))
input_var.cpu().numpy().astype('float32').tofile('testinput')
a = input_var.numpy()
filter_var = Variable(torch.randn(1, 1, 5, 5, 5))
filter_var.cpu().numpy().astype('float32').tofile('testfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(1))
biases_var.cpu().numpy().astype('float32').tofile('testbiases')
b = biases_var.numpy()
output_var = nn.functional.conv_transpose3d(input_var,filter_var,bias=biases_var)
output_var.cpu().numpy().astype('float32').tofile('testoutput')
print(output_var.shape)

run_mean = 5
run_var = 1
gamma = 1
beta = 0.5
bnormpar = torch.tensor([run_mean, run_var, gamma, beta])

bnormpar.numpy().astype('float32').tofile('bnormparams');

normed_var = nn.functional.batch_norm(output_var,bnormpar[0].reshape(1),
                                      bnormpar[1].reshape(1),
                                      weight=bnormpar[2].reshape(1),
                                      bias=bnormpar[3].reshape(1))

normed_var.cpu().numpy().astype('float32').tofile('testnormoutput')


normedrelu_var = nn.functional.relu(normed_var)
normedrelu_var.cpu().numpy().astype('float32').tofile('testnormreluoutput')
print(normedrelu_var.shape)
output = output_var.numpy()

