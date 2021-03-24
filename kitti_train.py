# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------
# This kitti_train.py was borrowed from leido's https://github.com/leido/pytorch-prednet
# And some code has been changed.
#------------------------------------------------------------------------------------------

import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from kitti_data import KITTI
from prednet import PredNet

import time
import logging



parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    metavar='LR',
    help='learning rate (default: 0.001)')
parser.add_argument(
    '--last-epoch',
    type=int,
    default=-1,
    help='set last epoch number if resume train (default: -1)')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--load-last',
    default=False,
    metavar='LL',
    help='load model dict and optimizer dict.')
parser.add_argument(
    '--env',
    default='prednet',
    metavar='ENV',
    help='any name (default: prednet)')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--nt',
    type=int,
    default=10,
    help='num of time steps (default: 10)')





args = parser.parse_args()
torch.manual_seed(args.seed)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

log = {}
setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(args.env))
d_args = vars(args)
for k in d_args.keys():
    log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))


num_epochs = 150
samples_per_epoch = 500  # 100
N_seq_val = 100  # number of sequences to use for validation
nt = args.nt # num of time steps
batch_size = 4 #16  If batch is 16, memory over 8G

# これが　下から　RGB３チャンネル　４８チャンネル　９６チャンネル　１９２チャネルの４層の定義かな
# conv の kernelは　すべて３で固定みたい
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)



if args.gpu_id >= 0:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())  # L_0
    #layer_loss_weights = Variable(torch.FloatTensor([[1.], [1.], [1.], [1.]]).cuda())  # L_all
    time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
    time_loss_weights[0] = 0
    time_loss_weights = Variable(time_loss_weights.cuda())
else:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]])) #.cuda())  # L_0
    #layer_loss_weights = Variable(torch.FloatTensor([[1.], [1.], [1.], [1.]])) #.cuda())  # L_all
    time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
    time_loss_weights[0] = 0
    time_loss_weights = Variable(time_loss_weights) #.cuda())

DATA_DIR = 'kitti_data'
TRAIN_DIR = 'trained'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')


if args.gpu_id >= 0:
    kitti_train = KITTI(train_file, train_sources, nt)
    kitti_val = KITTI(val_file, val_sources, nt, N_seq=N_seq_val)
else:
    kitti_train = KITTI(train_file, train_sources, nt, N_seq=500)
    print ('WARNNING:  possible_starts is set to 500. Data volume is short for train.')  # if not limit 500, memory will be over > 8G...
    kitti_val = KITTI(val_file, val_sources, nt, N_seq=N_seq_val)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True)


model = PredNet(R_channels, A_channels, output_mode='error', gpu_id=args.gpu_id)


if args.load_last:
    saved_state = torch.load( os.path.join(TRAIN_DIR, 'training-last.pt'), map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)
    print ('+load training-last.pt')


if args.gpu_id >= 0 and torch.cuda.is_available():
    print(' Using GPU.')
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_maker  = lr_scheduler.StepLR(optimizer = optimizer, step_size = 30, gamma = 0.5, last_epoch=args.last_epoch)  # adjust it!



if args.load_last:
    saved_state = torch.load( os.path.join(TRAIN_DIR, 'training-last-opt.pt'), map_location=lambda storage, loc: storage)
    optimizer.load_state_dict(saved_state)
    print ('+load training-last-opt.pt')



start_time = time.time()

model.train() # need ?

if args.last_epoch == -1:
    start_epoch =1  # normal
else:
    start_epoch =args.last_epoch  # resume start

for epoch in range(start_epoch, num_epochs+1):
    
    #
    epoch_mean=0.
    c0=0
    for i, inputs in enumerate(train_loader):
        # epoch 毎に　samples_per_epoch回だけまわす
        if i >= samples_per_epoch:
            break
        # Here, transpose was
        #inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
        
        if args.gpu_id >= 0:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs) #.cuda())
        
        errors = model(inputs) # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(torch.abs(errors))  # 念のため、absを追加してみた

        optimizer.zero_grad()

        errors.backward()

        optimizer.step()
        
        epoch_mean +=errors.item()
        c0 +=1
        
        if args.gpu_id >= 0:
            if (i+1)%50 == 0:  #i%10 == 0:
                #print('Epoch: {}/{}, step: {}/{}, errors: {}'.format(epoch, num_epochs, i, len(kitti_train)//batch_size, errors.item()))
                #print('Epoch: {}/{}, step: {}/{} '.format(epoch, num_epochs, i, len(kitti_train)//batch_size,))
                
                log['{}_log'.format(args.env)].info(
                    "Time {0}, Epoch {1} / {2}, step {3} / {4} , error, {5:.8f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        epoch, num_epochs, i+1, min([len(kitti_train), batch_size * samples_per_epoch] )//batch_size, errors.item() ))
        
        
        else:
            # when cpu
            if (i+1)%1 == 0:  #i%10 == 0:
                #print('Epoch: {}/{}, step: {}/{}, errors: {}'.format(epoch, num_epochs, i, len(kitti_train)//batch_size, errors.item()))
                #print('Epoch: {}/{}, step: {}/{} '.format(epoch, num_epochs, i, len(kitti_train)//batch_size,))
                
                log['{}_log'.format(args.env)].info(
                    "Time {0}, Epoch {1} / {2}, step {3} / {4} , error, {5:.8f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        epoch, num_epochs, i+1, len(kitti_train)//batch_size, errors.item() ))
            
            if i%1 == 0:  #i%10 == 0:
                torch.save(model.state_dict(), os.path.join(TRAIN_DIR, 'training-last.pt') )
                torch.save(optimizer.state_dict(), os.path.join(TRAIN_DIR, 'training-last-opt.pt'))
                print ('-save training-last')
            
            

    # save model every epoch
    torch.save(model.state_dict(), os.path.join(TRAIN_DIR, 'training-last.pt') )
    torch.save(optimizer.state_dict(), os.path.join(TRAIN_DIR, 'training-last-opt.pt'))
    print ('-save training-last')
    
    if epoch%10 ==0:
        torch.save(model.state_dict(), os.path.join(TRAIN_DIR, 'training-last_' + str(epoch) + '.pt') )
        torch.save(optimizer.state_dict(), os.path.join(TRAIN_DIR, 'training-last-opt_' + str(epoch) + '.pt'))
    
    # eval val every epoch
    with torch.no_grad():  # without save parameters, only forward
        val_epoch_mean=0.
        val_c0=0
        for i, inputs in enumerate(val_loader):
            if args.gpu_id >= 0:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs) #.cuda())
            
            
            errors = model(inputs) # batch x n_layers x nt
            loc_batch = errors.size(0)
            errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
            errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
            errors = torch.mean(errors)
            val_epoch_mean +=errors.item()
            val_c0 +=1
    
    # show every epoch error mean
    log['{}_log'.format(args.env)].info(
                    "Time {0}, Epoch {1} / {2}, epoch error mean, {3:.8f}, val error mean, {4:.8f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        epoch, num_epochs, (epoch_mean / c0), (val_epoch_mean / val_c0) ))
    # update lr
    lr_maker.step()

torch.save(model.state_dict(), os.path.join(TRAIN_DIR, 'training.pt'))
print ('train finish')


