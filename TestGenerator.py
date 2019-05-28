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



inputspad = np.fromfile('data/SPAD/batch_0/conv3/ds3out',dtype='float32')
input_var = Variable(torch.tensor(inputspad)).reshape(1,1,128,8,8)
filters = np.fromfile('data/SPAD/batch_0/conv3/conv3.0.weight',dtype='float32')
filter_var = Variable(torch.tensor(filters)).reshape(16,1,3,3,3)
biases = np.fromfile('data/SPAD/batch_0/conv3/conv3.0.bias',dtype='float32')
biases_var = Variable(torch.tensor(biases)).reshape(16)
bnorm_mean = np.fromfile('data/SPAD/batch_0/conv3/conv3.1.running_mean',dtype='float32')
bnorm_var = np.fromfile('data/SPAD/batch_0/conv3/conv3.1.running_var',dtype='float32')
bnorm_w = np.fromfile('data/SPAD/batch_0/conv3/conv3.1.weight',dtype='float32')
bnorm_b = np.fromfile('data/SPAD/batch_0/conv3/conv3.1.bias',dtype='float32')

bnormpar = torch.tensor([bnorm_mean, bnorm_var, bnorm_w, bnorm_b])

output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var, padding = 1)

normed_var = nn.functional.batch_norm(output_var,bnormpar[0],
                                      bnormpar[1],
                                      weight=bnormpar[2],
                                      bias=bnormpar[3])

normedrelu_var = nn.functional.relu(normed_var)
normedrelu_var.numpy().astype('float32').tofile('data/SPAD/batch_0/conv3/conv30out')

#######################################################################
input_var = Variable(torch.randn(1, 1, 32, 32, 32))
input_var.cpu().numpy().astype('float32').tofile('testinput')
a = input_var.numpy()
filter_var = Variable(torch.randn(1, 1, 3, 3, 3))
filter_var.cpu().numpy().astype('float32').tofile('testfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(1))
biases_var.cpu().numpy().astype('float32').tofile('testbiases')
b = biases_var.numpy()
output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var, padding = 1)
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

output = normedrelu_var.numpy()
oraw= output_var.numpy()



input_var = Variable(torch.randn(1, 10, 128, 32, 32))
input_var.cpu().numpy().astype('float32').tofile('testinput')
a = input_var.numpy()
filter_var = Variable(torch.randn(10, 10, 5, 5, 5))
filter_var.cpu().numpy().astype('float32').tofile('testfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(10))
biases_var.cpu().numpy().astype('float32').tofile('testbiases')
b = biases_var.numpy()
output_var = nn.functional.conv3d(input_var,filter_var,bias=biases_var)
output_var.cpu().numpy().astype('float32').tofile('testoutput')

output = output_var.numpy()


###### CONV TRANSPOSE #########
input_var = Variable(torch.randn(1, 1, 8, 8, 8))
input_var.cpu().numpy().astype('float32').tofile('transtestin')
a = input_var.numpy()
filter_var = Variable(torch.randn(1, 1, 6, 6, 6))
filter_var.cpu().numpy().astype('float32').tofile('transtestfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(1))
biases_var.cpu().numpy().astype('float32').tofile('transtestbiases')
b = biases_var.numpy()
output_var = nn.functional.conv_transpose3d(input_var,filter_var,bias=biases_var, stride=2, padding=2)
output_var.cpu().numpy().astype('float32').tofile('transtestoutput')
run_mean = 5
run_var = 1
gamma = 1
beta = 0.5
bnormpar = torch.tensor([run_mean, run_var, gamma, beta])

bnormpar.numpy().astype('float32').tofile('transbnormparams');

normed_var = nn.functional.batch_norm(output_var,bnormpar[0].reshape(1),
                                      bnormpar[1].reshape(1),
                                      weight=bnormpar[2].reshape(1),
                                      bias=bnormpar[3].reshape(1))

normed_var.cpu().numpy().astype('float32').tofile('transtestnormoutput')


normedrelu_var = nn.functional.relu(normed_var)
normedrelu_var.cpu().numpy().astype('float32').tofile('transtestnormreluoutput')

output = output_var.numpy()


###### CONV TRANSPOSE #########
input_var = Variable(torch.randn(1, 2, 128, 8, 8))
input_var.cpu().numpy().astype('float32').tofile('transtestin')
a = input_var.numpy()
filter_var = Variable(torch.randn(2, 2, 6, 6, 6))
filter_var.cpu().numpy().astype('float32').tofile('transtestfilters')
w = filter_var.numpy()

biases_var = Variable(torch.ones(2))
biases_var.cpu().numpy().astype('float32').tofile('transtestbiases')
b = biases_var.numpy()
output_var = nn.functional.conv_transpose3d(input_var,filter_var,bias=biases_var, stride=2, padding=2)
output = output_var.numpy()
#output_var.cpu().numpy().astype('float32').tofile('transtestoutput')
run_mean = [5,5]
run_var = [1,1]
gamma = [1,1]
beta = [0.5,0.5]
bnormpar = torch.tensor([run_mean, run_var, gamma, beta])

bnormpar.numpy().astype('float32').tofile('transbnormparams');

normed_var = nn.functional.batch_norm(output_var,bnormpar[0],
                                      bnormpar[1],
                                      weight=bnormpar[2],
                                      bias=bnormpar[3])

#normed_var.cpu().numpy().astype('float32').tofile('transtestnormoutput')


normedrelu_var = nn.functional.relu(normed_var)
normedrelu_var.cpu().numpy().astype('float32').tofile('transtestnormreluoutput')




