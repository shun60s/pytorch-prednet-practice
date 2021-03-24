# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------
# This kitti_data.py was borrowed from leido's https://github.com/leido/pytorch-prednet
# And some code has been changed.
#------------------------------------------------------------------------------------------

import hickle as hkl

import torch
import torch.utils.data as data
import numpy as np



class KITTI(data.Dataset):
    def __init__(self, datafile, sourcefile, nt, shuffle=False, sequence_start_mode='all', N_seq=None):
        self.datafile = datafile
        self.sourcefile = sourcefile
        self.X = hkl.load(self.datafile)
        self.sources = hkl.load(self.sourcefile)
        self.nt = nt
        cur_loc = 0
        possible_starts = []
        
        
        print ('X.shape', self.X.shape)  # train (41396, 128, 160, 3)     val (154,128,160,3) 
                                         #        num    h    w   colors    ->  num colors h w
        print ('sources.shape', self.sources.shape) # train (41396,)        val  (154,)
        
        self.X = self.X.transpose(0,3,1,2)
        print ('X.shape after transpose', self.X.shape)
        
        self.sequence_start_mode = sequence_start_mode
        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            while cur_loc < self.X.shape[0] - self.nt + 1:
                if self.sources[cur_loc] == self.sources[cur_loc + self.nt - 1]:
                    possible_starts.append(cur_loc)
                    cur_loc += self.nt
                    
                else:
                    cur_loc += 1
            
            self.possible_starts = possible_starts
            
            
        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        
        print ('len(self.possible_starts)',len(self.possible_starts))
        #print (self.X[0].astype(np.float32)/255)

    def __getitem__(self, index):
        loc = self.possible_starts[index]
        #print ('index, loc', index, loc)
        #return self.X[loc:loc+self.nt]  # 1st index is used
        return torch.from_numpy( self.X[loc:loc+self.nt].astype(np.float32)/255)     # 1st index is used


    def __len__(self):
        return len(self.possible_starts)