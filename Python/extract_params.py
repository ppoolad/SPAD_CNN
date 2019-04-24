# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:37:29 2019

@author: ppool
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import configparser
from configparser import ConfigParser
from models import FusionDenoiseModel, DenoiseModel, Upsample8xDenoiseModel
import scipy
import scipy.io
import os
import pathlib


cudnn.benchmark = True
dtype = torch.cuda.FloatTensor#torch.cuda.FloatTensor

parser = argparse.ArgumentParser(
        description='PyTorch Deep Sensor Fusion Middlebury Evaluation')
parser.add_argument('--option', default=None, type=str,
                    metavar='NAME', help='Name of model to use with options in config file, \
                    either FusionDenoise, Denoise, or Upsample8xDenoise)')
parser.add_argument('--config', default='middlebury.ini', type=str,
                    metavar='FILE', help='name of configuration file')
parser.add_argument('--gpu', default=None, metavar='N',
                    help='which gpu')
parser.add_argument('--ckpt_noise_param_idx', nargs='+', default=None,
                    metavar='N', type=str,
                    help='which noise level we are evaluating on \
                         (value 1-9, default: all)')
parser.add_argument('--scene', default=None, type=str, nargs='+',
                    metavar='FILE', help='name of scene to use \
                    (default: NONE->all)')
parser.add_argument('--naive', default=None, type=str,
                    metavar='1 or 0', help='If option is Upsample8xDenoise \
                    then enable naive upsample with pretrained weights')

simulation_params = [[10, 2], [5, 2], [2, 2],
                     [10, 10], [5, 10], [2, 10],
                     [10, 50], [5, 50], [2, 50]]
scenedir = 'middlebury/processed/'
outdir = 'results_middlebury/'
scenenames = ['Art', 'Books', 'Bowling1', 'Dolls',
              'Laundry', 'Moebius', 'Plastic', 'Reindeer']
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)


def parse_arguments(args):
    config = ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.optionxform = str
    config.read(args.config)

    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')

    if args.gpu:
        config.set('params', 'gpu', args.gpu)
    if args.ckpt_noise_param_idx:
        config.set('params', 'ckpt_noise_param_idx',
                   ' '.join(args.ckpt_noise_param_idx))
    if args.scene:
        config.set('params', 'scene', ' '.join(args.scene))
    if args.naive:
        config.set('Upsample8xDenoise', 'naive', args.naive)

    # read all values from config file
    opt = {}
    opt['gpu'] = config.get('params', 'gpu')

    opt['ckpt_noise_param_idx'] = \
        config.get('params', 'ckpt_noise_param_idx').split()
    opt['ckpt_noise_param_idx'] = \
        [int(idx) for idx in opt['ckpt_noise_param_idx']]
    if not opt['ckpt_noise_param_idx']:
        opt['ckpt_noise_param_idx'] = np.arange(1, 11)

    opt['option'] = config.get('params', 'option')
    opt['scene'] = config.get('params', 'scene').split()
    if not opt['scene']:
        opt['scene'] = scenenames
    opt['scenesizes'] = dict(config.items('SceneSizes'))
    opt['checkpoint'] = []

    if option != 'Upsample8xDenoise':
        for n in opt['ckpt_noise_param_idx']:
            opt['checkpoint'].append(config.get(option,
                                     'ckpt_noise_param_{}'.format(n)))
    else:
        opt['naive'] = int(config.get('Upsample8xDenoise', 'naive'))
        if opt['naive']:
            opt['checkpoint_msgnet'] = config.get(option, 'ckpt_msgnet')
            opt['checkpoint'].append(config.get(option, 'ckpt_noise_param_10'))
        else:
            opt['checkpoint'].append(
                config.get(option, 'ckpt_finetune_noise_param_10'))
    return opt


def process_denoise(opt, model, middlebury_filename, out_filename):
    intensity = scipy.io.loadmat(middlebury_filename)['intensity']
    intensity = np.asarray(intensity).astype(np.float32)
    s1, s2 = intensity.shape

    spad = scipy.io.loadmat(middlebury_filename)['spad']
    spad = scipy.sparse.csc_matrix.todense(spad)
    spad = np.asarray(spad).reshape([s2, s1, -1])

    intensity = torch.from_numpy(intensity).type(dtype)
    intensity = intensity.unsqueeze(0).unsqueeze(0)
    intensity_var = Variable(intensity)
    spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    spad = spad.unsqueeze(0).unsqueeze(0)
    spad_var = Variable(spad.type(dtype))

    batchsize = 2
    dim1 = 64
    dim2 = 64
    step = 32
    num_rows = int(np.floor((s1 - dim1)/step + 1))
    num_cols = int(np.floor((s2 - dim2)/step + 1))
    im = np.zeros((s1, s2))
    smax_im = np.zeros((s1, s2))
    for i in tqdm(range(num_rows)):
        for j in range(0, num_cols, batchsize):
            # set dimensions
            begin_idx = step//2
            end_idx = dim1 - step//2
            b_idx = 0
            for k in range(batchsize):
                test = s2 - ((j+k)*step + dim2)
                if test >= 0:
                    b_idx += 1
            iter_batchsize = b_idx

            sp1 = Variable(torch.zeros(iter_batchsize,
                                       1, 1024, dim1, dim2))
            i1 = Variable(torch.zeros(iter_batchsize,
                                      1, dim1, dim2))
            for k in range(iter_batchsize):
                sp1[k, :, :, :, :] = spad_var[:, :, :, i*step:(i)*step + dim1,
                                              (j+k)*step:(j+k)*step + dim2]
                i1[k, :, :, :] = intensity_var[:, :, i*step:(i)*step + dim1,
                                               (j+k)*step:(j+k)*step + dim2]

            if opt['option'] == 'FusionDenoise':
                denoise_out, sargmax = model(sp1.type(dtype), i1.type(dtype))
            else:
                denoise_out, sargmax = model(sp1.type(dtype))
            denoise = np.argmax(denoise_out.data.cpu().numpy(), axis=1)
            denoise = denoise.squeeze()
            smax = sargmax.data.cpu().numpy().squeeze() * 1024

            for k in range(sp1.shape[0]):
                im[i*step:(i+1)*step, (j+k)*step:(j+k+1)*step] = \
                    denoise[k, begin_idx:end_idx, begin_idx:end_idx].squeeze()
                smax_im[i*step:(i+1)*step, (j+k)*step:(j+k+1)*step] = \
                    smax[k, begin_idx:end_idx, begin_idx:end_idx].squeeze()

    out = {'im': im, 'smax': smax_im}
    scipy.io.savemat(out_filename, out)

def main():
    args = parser.parse_args()
    opt = parse_arguments(args)

    # set gpu
    print('=> setting gpu to {}'.format(opt['gpu']))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    # print options
    print('=> Running scenes {}'.format(', '.join(opt['scene'])))
    model_str = [str(idx) for idx in opt['ckpt_noise_param_idx']]
    print('=> for models trained on noise levels {}'.format(', '
          .join(model_str)))

    # iterate over trained models
    for model_iter, model_param in enumerate(opt['ckpt_noise_param_idx']):
        print('=> Initializing Model')
        model = eval(opt['option'] + 'Model()')
        model.type(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print('=> Loading checkpoint {}'.format(opt['checkpoint'][model_iter]))
        ckpt = torch.load(opt['checkpoint'][model_iter],map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        try:
            ckpt_dict = ckpt['state_dict']
        except KeyError:
            print('Key error loading state_dict from checkpoint; assuming \
checkpoint contains only the state_dict')
            ckpt_dict = ckpt
        dirname = 'Mode_parameters/'
        for k in ckpt_dict.keys():
            # by pooya to extract data #
            data = ckpt_dict[k]
            data.numpy().astype('float32').tofile(dirname+k)
            
            ############################
            model_dict.update({k: ckpt_dict[k]})
            
if __name__ == '__main__':
    main()