# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------
# This prednet.py was borrowed from leido's https://github.com/leido/pytorch-prednet
# And some code has been changed.
#------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell
from torch.autograd import Variable


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error', gpu_id=-1):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience via i+1 (3, 48, 96, 192, 0)# representation neurons (R_channels = (3, 48, 96, 192)+(0,)
        self.a_channels = A_channels          # layer-specific prediction A_channels = (3, 48, 96, 192)
        self.n_layers = len(R_channels)       # 4 = len(3, 48, 96, 192)
        self.output_mode = output_mode
        self.gpu_id = gpu_id
        self.extrap_start_time=None

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        # 
        for i in range(self.n_layers):
            # in_channels, out_channels?, kernel_size(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True):
            #   入力はrepresentation neurons[i+1] とan error term＝　layer-specific prediction +/- target
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i], (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            # hut_A[l] input representation neurons output a_channels[i]
            # sequential 形式で定義している
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            
            # chnage to use clamp
            #if i == 0:  # はじめ層だけ　SatLU()　を追加している
            #    conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        
        # A[l+1] 用の max poolingの定義　スライドが２で大きさ半分か
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            # A[l+1] input error term[l]   output a_channels[l+1]
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        # ConvLSTMCellのreset
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input):

        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)
        
        # 初期値を０に設定する
        for l in range(self.n_layers):
            if self.gpu_id >= 0:
                E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h)).cuda()
                R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()
            else:
                E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h)) #.cuda()
                R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)) #.cuda()
            # 1層経るたびに、大きさが1/2になる
            w = w//2
            h = h//2
            
        time_steps = input.size(1)
        total_error = []
        
        
        # add A_hat_stack
        if self.output_mode == 'prediction':
            A_hat_stack = torch.zeros_like(input)
        else:
            A_hat_stack = None
        
        # time_steps回　繰り返す
        for t in range(time_steps):
            
            A = input[:,t]  # t番目の生データをセット
            if self.gpu_id >= 0:
                A = A.type(torch.cuda.FloatTensor)
            else:
                A = A.type(torch.FloatTensor)  #A.type(torch.cuda.FloatTensor)
            
            
            # convLSTMの計算
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                
                # highest layer 
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                
                R_seq[l] = R
                H_seq[l] = hx

            # hut Aの計算
            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    # chnage to use clamp
                    A_hat = torch.clamp(A_hat, 0, 1) 
                    frame_prediction = A_hat  # lowest layer 0 prediction output
                    
                    # add A_hat_stack
                    if self.output_mode == 'prediction':
                        A_hat_stack[:,t] = A_hat
                    
                    
                    # extrap_start_timeを超えたら入力をA_hatに置き換える
                    if (self.extrap_start_time is not None) and t >= self.extrap_start_time:
                        A= A_hat
                        #print (' extrap t',t)
                
                #　error の計算　Atlowest layer 0, pos= prediction output -  input raw 
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                
                # highest layer 以外は　ここで　上位のAを計算する。大きさ半分か
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
            
            
            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)



        if self.output_mode == 'error':
            return torch.stack(total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            #return frame_prediction
            return A_hat_stack

    def set_extrap_start_time(self, extrap_start_time = None):
        self.extrap_start_time=extrap_start_time
