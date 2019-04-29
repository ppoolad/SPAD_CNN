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



inputspad = np.fromfile('data/test/batch_0/conv3d/spadfile',dtype='float32')
input_var = Variable(torch.tensor(inputspad)).reshape(1,1,1024,64,64)
filters = np.fromfile('data/test/batch_0/conv3d/conv0.0.weight',dtype='float32')
filter_var = Variable(torch.tensor(filters)).reshape(4,1,9,9,9)
biases = np.fromfile('data/test/batch_0/conv3d/conv0.0.bias',dtype='float32')
biases_var = Variable(torch.tensor(biases)).reshape(4)
bnorm_mean = np.fromfile('data/test/batch_0/conv3d/conv0.1.running_mean',dtype='float32')
bnorm_var = np.fromfile('data/test/batch_0/conv3d/conv0.1.running_var',dtype='float32')
bnorm_w = np.fromfile('data/test/batch_0/conv3d/conv0.1.weight',dtype='float32')
bnorm_b = np.fromfile('data/test/batch_0/conv3d/conv0.1.bias',dtype='float32')

bnormpar = torch.tensor([bnorm_mean, bnorm_var, bnorm_w, bnorm_b])

output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var, padding=4)

normed_var = nn.functional.batch_norm(output_var,bnormpar[0],
                                      bnormpar[1],
                                      weight=bnormpar[2],
                                      bias=bnormpar[3])

normedrelu_var = nn.functional.relu(normed_var)
normedrelu_var.numpy().astype('float32').tofile('conv0out')


input_var = Variable(torch.randn(1, 1, 1024, 64, 64))
input_var.cpu().numpy().astype('float32').tofile('testinput')
a = input_var.numpy()
filter_var = Variable(torch.randn(1, 1, 5, 5, 5))
filter_var.cpu().numpy().astype('float32').tofile('testfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(1))
biases_var.cpu().numpy().astype('float32').tofile('testbiases')
b = biases_var.numpy()
output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var)
output_var.cpu().numpy().astype('float32').tofile('testoutput')
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

output = output_var.numpy()




input_var = Variable(torch.randn(1, 10, 1024, 64, 64))
input_var.cpu().numpy().astype('float32').tofile('testinput')
a = input_var.numpy()
filter_var = Variable(torch.randn(1, 10, 5, 5, 5))
filter_var.cpu().numpy().astype('float32').tofile('testfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(1))
biases_var.cpu().numpy().astype('float32').tofile('testbiases')
b = biases_var.numpy()
output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var)
output_var.cpu().numpy().astype('float32').tofile('testoutput')

output = output_var.numpy()