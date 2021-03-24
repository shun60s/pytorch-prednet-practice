# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------
# This kitti_test.py was borrowed from leido's https://github.com/leido/pytorch-prednet
# And some code has been changed.
#------------------------------------------------------------------------------------------


import torch
import os
import argparse
import numpy as np
import hickle as hkl

from torch.utils.data import DataLoader
from torch.autograd import Variable
from kitti_data import KITTI
from prednet import PredNet

import torchvision
from PIL import Image


parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--nt',
    type=int,
    default=10,
    help='num of time steps (default: 10)')
parser.add_argument(
    '--et',
    type=int,
    default=-1,
    help='extrap_start_time (default:-1 means off)')

args = parser.parse_args()

batch_size = 4 # 2
nt = args.nt # num of time steps
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

DATA_DIR = 'kitti_data'
TRAIN_DIR = 'trained'
RESULTS_DIR = 'results'

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

kitti_test = KITTI(test_file, test_sources, nt, sequence_start_mode='unique')
test_loader = DataLoader(kitti_test, batch_size=batch_size, shuffle=False)

model = PredNet(R_channels, A_channels, output_mode='prediction', gpu_id=args.gpu_id)
model.load_state_dict(torch.load( os.path.join(TRAIN_DIR, 'training.pt'), map_location=lambda storage, loc: storage ))

if args.et > 0 and args.et < args.nt:
    model.set_extrap_start_time( extrap_start_time=args.et)

if args.gpu_id >= 0 and torch.cuda.is_available():
    print(' Using GPU.')
    model.cuda()

c0=0
for i, inputs in enumerate(test_loader):
    #
    with torch.no_grad():  # without save parameters, only forward
        if args.gpu_id >= 0:
            inputs = Variable(inputs.cuda())
            inputs2 = inputs * 255
            origin = inputs2.data.cpu().byte() #[:, nt-1]  # set last data
        else:
            inputs = Variable(inputs) #.cuda())
            inputs2 = inputs * 255
            origin = inputs2.data.byte() #[:, nt-1]  # cpu().byte()[:, nt-1]  batch x latest time stemp=9(10-1)
        
        """
        print(' origin:')
        print(type(origin))
        print(origin.size())
        """
        pred = model(inputs)
        pred2 = pred * 255
        
        if args.gpu_id >= 0:
            pred = pred2.data.cpu().byte()
        else:
            pred = pred2.data.byte()
        
        for l in range(batch_size):
            origin2 = torchvision.utils.make_grid(origin[l], nrow=nt)
            pred2 = torchvision.utils.make_grid(pred[l], nrow=nt)
            
            im1 = Image.fromarray(np.rollaxis(origin2.numpy(), 0, 3))
            im2 = Image.fromarray(np.rollaxis(pred2.numpy(), 0, 3))
            dst = Image.new('RGB', (im1.width, im1.height + im2.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            
            if args.et > 0 and args.et < args.nt:
                save_file_name = os.path.join(RESULTS_DIR,'origin_vs_predicted_' + str(c0) + '-extrap_start' + str(args.et) + '.jpg')
            else:
                save_file_name = os.path.join(RESULTS_DIR,'origin_vs_predicted_' + str(c0)+ '.jpg')
            
            dst.save( save_file_name )
            print ('save to ', save_file_name)
            c0 +=1
            
            # When cpu, only two times
            if not (args.gpu_id >= 0) and c0 >= 2:
                break
        
    # When cpu, only two times
    if not (args.gpu_id >= 0) and c0 >= 2:
        break

