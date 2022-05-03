import os 
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
import ConvLSTM

def test(args):

    
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    gpu = args.gpu_id if torch.cuda.is_available() else None

    if args.model_name.split('_')[-1] == 'tf':
        teacher_forcing_rate = 0
    else:
        teacher_forcing_rate = None
        
    # [TODO: load testset and model]
    if args.data_source == 'moving_mnist':
        from moving_mnist import MovingMNIST
        train_dataset = MovingMNIST(root=args.data_path,
                                    is_train=False, 
                                    seq_len=args.seq_len, 
                                    horizon=args.horizon)
    
    elif args.data_source == 'sst':
        from sst import SST
        train_dataset = SST(root=args.data_path,
                            is_train=False, 
                            seq_len=args.seq_len, 
                            horizon=args.horizon)


    num_samples = 0
    avg_mse = 0
    avg_ssim = 0
    frame_mse = [0] * args.horizon
    frame_ssim = [0] * args.horizon
    args.h_channels = 64
    args.in_channels = 1

    ground_truth, prediction = [], []  # these are for visualization
    model = ConvLSTM.Seq2Seq(args)
    model.load_state_dict(torch.load(args.model_name))
    model.to(device)
    data_loader = DataLoader(
        train_dataset, 
        shuffle=True)
    batch_size = 1
    model.eval()    
    with torch.no_grad():
        for it, (data,target) in enumerate(data_loader):
            inputs, labels = data.to(gpu), target.to(gpu)
            inputs = inputs.float()
            outputs = model(inputs,labels,teacher_forcing_rate)
            num_samples+=len(data)
            frames = labels
            out = outputs
            # [TODO: generate predicted frames]

            # num_samples += [batch size]
            # frames = [ground truth frames]
            # out = [predicted frames]
            
            frames = frames.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            
            frames = frames[:, -args.horizon:, ...]
            out = out[:, -args.horizon:, ...]            
             
            if num_samples < 100:
                ground_truth.append(frames)
                prediction.append(out)

            for i in range(args.horizon):
                
                frame_i = frames[:, i, ...]
                out_i = out[:, i, ...]
                mse = np.square(frame_i - out_i).sum()

                ssim = 0
                for b in range(batch_size):
                    ssim += compare_ssim(frame_i[b, 0], out_i[b, 0])
                
                frame_mse[i] += mse
                frame_ssim[i] += ssim
                avg_mse += mse
                avg_ssim += ssim
                                
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)
            
    avg_mse = avg_mse / (num_samples * args.horizon)
    avg_ssim = avg_ssim / (num_samples * args.horizon)
    
    print('mse: {:.4f}, ssim: {:.4f}'.format(avg_mse, avg_ssim))
    for i in range(args.horizon):
        print('frame {} - mse: {:.4f}, ssim: {:.4f}'.format(i + args.seq_len + 1, 
                                                            frame_mse[i] / num_samples, 
                                                            frame_ssim[i] / num_samples))
        
    np.savez_compressed(
        os.path.join('{}/{}_{}_prediction.npz'.format(args.result_path, args.model_name, args.data_source)),
        ground_truth=ground_truth,
        prediction=prediction)


def main():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    parser.add_argument('--model_name', type=str, required=True, help='the name of model')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to the datasets')
    parser.add_argument('--data_source', type=str, required=True, help='moving_mnist | sst')
    parser.add_argument('--result_path', type=str, default='./results/', help='path to the results, containing model and logs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seq_len', type=int, required=True, help='input frame length')
    parser.add_argument('--horizon', type=int, required=True, help='output frame length')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)    
    
    args = parser.parse_args()
    test(args)
    
#python test.py --model_name ../results/mnist_adam12.pth --data_path ../data/ --data_source moving_mnist --result_path ../results/ --seq_len 10 --horizon 10
#python Orpheus/code/test.py --model_name sst_adam53.pth --data_path drive/MyDrive/assignment5/data/ --data_source moving_mnist --result_path results/ --seq_len 10 --horizon 10    
if __name__ == "__main__":
    main()
