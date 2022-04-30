from __future__ import print_function
import torch.utils.data as data

import os
import numpy as np
import torch


class MovingMNIST(data.Dataset):
    
    def __init__(
        self, 
        root,
        is_train=True, 
        seq_len=10,
        horizon=10,
    ):
        
        self.is_train = is_train
        self.seq_len = seq_len
        self.horizon = horizon

        if self.is_train:
            path = os.path.join(root, 'moving_mnist_train.npy')
            self.data = np.load(path)
        else:
            path = os.path.join(root, 'moving_mnist_test.npy')
            self.data = np.load(path)
        
    def __getitem__(self, index):
        
        input_seq = self.data[index, :self.seq_len]
        output_seq = self.data[index, self.seq_len: (self.seq_len + self.horizon)]

        input_seq = torch.from_numpy(input_seq / 255.0).contiguous().float()   
        output_seq = torch.from_numpy(output_seq / 255.0).contiguous().float()
        return input_seq, output_seq

    def __len__(self):
        return self.data.shape[0]
