import os 
import json
import time
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ConvLSTM

def train(args):
    
    gpu = args.gpu_id if torch.cuda.is_available() else None
    print(f'Device - GPU:{args.gpu_id}')
    
    if args.data_source == 'moving_mnist':
        from moving_mnist import MovingMNIST
        train_dataset = MovingMNIST(root=args.data_path,
                                    is_train=True, 
                                    seq_len=args.seq_len, 
                                    horizon=args.horizon)
        
    elif args.data_source == 'sst':
        from sst import SST
        train_dataset = SST(root=args.data_path,
                            is_train=True, 
                            seq_len=args.seq_len, 
                            horizon=args.horizon)
                    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True)
    
    # [TODO: define your model, optimizer, loss function]
    args.h_channels = 64
    args.in_channels = 1
    model = ConvLSTM.Seq2Seq(args)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-3, lr = 0.001, momentum=0.9)
    criterion = nn.MSELoss()
    writer = SummaryWriter()  # initialize tensorboard writer
    
    if args.use_teacher_forcing:
        teacher_forcing_rate = 1.0
    else:
        teacher_forcing_rate = None
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0         

        for i, (data,target) in enumerate(train_loader):
            print(data.shape)
            inputs, labels = data.to(gpu), target.to(gpu)
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs,labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            writer.add_scalar()
        print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, running_loss/len(train_set)))
        running_loss = 0.0
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'ep_' + str(epoch) +'.pth'))

            # [TODO: train the model with a batch]
            
            # [TODO: use tensorboard to visualize training loss]
            
            
            # [TODO: if using teacher forcing, update teacher forcing rate]

    
    
def main():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    # load data from file
    parser.add_argument('--data_path', type=str, default='./data/', help='path to the datasets')
    parser.add_argument('--data_source', type=str, required=True, help='moving_mnist | sst')
    parser.add_argument('--model_name', type=str, required=True, help='name of the saved model')
    parser.add_argument('--seq_len', type=int, required=True, help='input frame length')
    parser.add_argument('--horizon', type=int, required=True, help='output frame length')
    parser.add_argument('--use_teacher_forcing', action='store_true', help='if using teacher forcing, default is False')
    parser.add_argument('--result_path', type=str, default='./results', help='path to the results, containing model and logs')

    # training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    
    train(args)
    
    
if __name__ == "__main__":
    main()
