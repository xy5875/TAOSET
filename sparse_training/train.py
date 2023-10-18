import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast as autocast


# 从所给的文件中导入相应的函数和类
from resnet import resnet50
import numpy as np

def setup_logging(args):
    logging.basicConfig(level=logging.INFO)
    if args.if_log:
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(args.log_file, mode='a', delay=True)  
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def get_args():
    
    parser = argparse.ArgumentParser(description='PyTorch SwiftTTA')
    
    # general parameters, dataloader parameters
    parser.add_argument('--mode', default='train', type=str, help='test or train')
    
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--seed', default=2023, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu_id', default=6, type=int, help='GPU id to use.')
    
    parser.add_argument('--if_log', default=True, type=bool, help='Whether to log to a file')
    parser.add_argument('--log_file', default='/taoset/Desktop/sparse_training/cifar10_0.001.log', type=str, help='Log file name')#jittor_grayscale
    parser.add_argument('--continue_training', default=False, help='Whether to continue training from a checkpoint')
    parser.add_argument('--ckpt_dir', default="/data/user21100736/fc/SwiftTTA/method/checkpoints/resnet50_gn_acc_0.8306_cifar10.pth", help='Whether to continue training from a checkpoint')

    
    # train settings 
    parser.add_argument('--train_data', default='nico', type=str, help='dataset kind')
    parser.add_argument('--train_data_dir', default='/data/user21100736/fc/DATASET/NICO++/track_1/track_1/public_dg_0416/train/outdoor', type=str, help='data directory')
    
    #parser.add_argument('--pretrained_dir', default='/data/user21100736/fc/SwiftTTA/method/checkpoints/resnet50_gn_imagenet.pth', type=str, help='pth directory')
    parser.add_argument('--pretrained_dir', default='/taoset/Desktop/sparse_training/resnet50_bn_imagenet.pth', type=str, help='pth directory')
    
    parser.add_argument('--checkpoints_dir_to_save', default='/data/user21100736/fc/SwiftTTA/method/checkpoints/nico/', type=str, help='pth directory')
    parser.add_argument('--train_epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    
    parser.add_argument('--layer_latency_path', default='/taoset/Desktop/sparse_training/resnet50_224_10_64_cpu_profile.pth', type=str, help='layer-wise latency file path')
    
    #test data
    parser.add_argument('--test_data', default='cifar10', type=str, help='dataset')
    parser.add_argument('--test_data_dir', default='/taoset/Desktop/sparse_training/', type=str, help='data directory')
    
    
    parser.add_argument('--batch_size', default=128, type=int, help='train and test data batch size')
    
    parser.add_argument('--speed_up_ratio', default=0.1, type=float, help='user-specified ratio of the full training time')
    parser.add_argument('--Tq', default=1000, type=int, help='scaling factor for DP search')
    
    parser.add_argument('--sparse_training', action='store_true', help='sparse training or not')
    parser.add_argument('--autocast', action='store_true', help='sparse training or not')
    

    return parser.parse_args()

def selection_DP(t_dy, t_dw, I, rho=0.3):
    """
    Solving layer selection problem via dynamic programming

    Args:
        t_dy (np.array int16): downscaled t_dy [N,]
        t_dw (np.array int16): downscaled t_dw [N,]
        I (np.array float32): per-layer contribution to loss drop [N,]
        rho (float32): backprop timesaving ratio
    """
    
    # Initialize the memo tables of subproblems
    N = t_dw.shape[0] # number of NN layers
    T = np.sum(t_dw + t_dy) # maximally possible BP time
    T_limit = int(rho * T)
    t_dy_cumsum = 0
    for k in range(N):
        t_dy_cumsum += t_dy[k]
        if t_dy_cumsum > T_limit:
            break
    N_limit = k
    # Infinite importance
    MINIMAL_IMPORTANCE = -99999.0
    # L[k, t] - maximum cumulative importance when:
    # 1. selectively training within last k layers,
    # 2. achieving BP time at most t
    L_memo = np.zeros(shape=(N_limit + 1, T_limit + 1), dtype=np.float32)
    L_memo[0, 0] = 0
    #L_memo[0, 1:] = MINIMAL_IMPORTANCE
        
    # M[k, t, :] - solution to subproblem L[k, t]
    M_memo = np.zeros(shape=(N_limit + 1, T_limit + 1, N), dtype=np.uint8)
    
    S_memo = np.zeros(shape=(N_limit + 1, T_limit + 1), dtype=np.uint8)
    S_memo[0, 0] = 1
    S_memo[1:, 0] = 1
    S_memo[0, 1:] = 1
    
    max_importance = MINIMAL_IMPORTANCE
    k_final, t_final = 0, 0
    # Solving all the subproblems recursively
    for k in range(1, N_limit + 1):
        for t in range(0, T_limit + 1):
            # Subproblem 1:
            # If layer k-1 is NOT selected
            # --> no BP time increase 
            # --> no importance increase
            l_skip_curr_layer = L_memo[k - 1, t]
            
            # Subproblem 2:
            # If layer k-1 is selected
            # --> BP time increases dt = t_dw[k - 1] + sum(t_dy[k-2 : n])
            opt_k = -1
            opt_t = -1
            l_max = l_skip_curr_layer
            t_p = t - t_dw[k - 1]
            # traverse from layer k-1 to the beginning
            for k_p in range(k - 1, -1, -1):
                t_p -= t_dy[k_p]
                if t_p >= 0 and S_memo[k_p, t_p] == 1:
                    l_candidate = L_memo[k_p, t_p] + I[k - 1]
                    if l_candidate > l_max:
                        opt_k = k_p
                        opt_t = t_p
                        l_max = l_candidate
                        
            # make sure valid solution found by checking integer variable
            if opt_k >= 0:
                L_memo[k, t] = l_max
                M_memo[k, t, :(k - 1)] = M_memo[opt_k, opt_t, :(k - 1)]
                M_memo[k, t, k - 1] = 1
                S_memo[k, t] = 1
            # no valid solution from backtrace or no larger than not selecting
            else:
                L_memo[k, t] = l_skip_curr_layer
                M_memo[k, t, :(k - 1)] = M_memo[k - 1, t, :(k - 1)]
                M_memo[k, t, k - 1] = 0
                S_memo[k, t] = 0
            
            if l_max > max_importance:
                max_importance = L_memo[k, t]
                k_final, t_final = k, t
    
    M_sol = M_memo[k_final, t_final, :]
    return max_importance, M_sol


def train(train_loader, test_loader, model, criterion, optimizer, device, highest_accuracy, args):
    #model.train()
    total_loss = 0.0
    
    iter = 0
    mean_time = 0
    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        start_time = time.time()
        
        if model.sparse_training == True:
            #完整反向传播
            if iter % 100 == 0:
                
                model.requires_grad_(True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                #评估层重要性
                for name, module in model.named_modules():
                    if name in model.param_module_names:
                        index = model.param_module_names.index(name)
                        weight_sum = module.weight.grad.sum()
                        if module.bias is not None:
                            bias_sum = module.bias.grad.sum()
                        else:
                            bias_sum = 0    
                        model.layer_importance[index] = (weight_sum + bias_sum).cpu().numpy()
                
                max_importance, model.selected_layers=selection_DP(model.t_dy_q, model.t_dw_q, model.layer_importance, model.speed_up_ratio)
                model.selected_layers = np.flip(model.selected_layers)
                
                model.requires_grad_(False)
                layer_id=0
                for name, m in model.named_modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.GroupNorm)):
                        if model.selected_layers[layer_id]==1:
                            m.requires_grad_(True)
                            #model.active_layers.append(name)
                        layer_id+=1
                
            else:
                #稀疏反向传播
                if model.autocast == True:
                    with autocast():
                        output = model(data)
                else:
                    output = model(data)
                loss = criterion(output, target)

                loss.backward()
                
            
        else:
            model.requires_grad_(True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
        '''dot = make_dot(loss, params=dict(model.named_parameters()))
        dot.render('/data/user21100736/fc/TAO-SET/algorithm/loss_computation_graph')'''
        optimizer.step()
        total_loss += loss.item()
        
        per_batch_time = time.time() - start_time
        mean_time += per_batch_time
        if iter % 2 == 0:
            print('iter: {}, loss: {:.3f}, throughout: {:.4f}, per_batch_time: {:.4f}, mean_time: {:.4f}'.format(iter, loss, len(data)/per_batch_time, per_batch_time, mean_time/(iter+1)))
        iter += 1
        
    avg_train_loss = total_loss / len(train_loader)

    # Validate the model on the test set to calculate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    # Save the model if the current accuracy is the highest so far
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        model_name = f"resnet50_nico_noaug_bn_acc_{accuracy:.4f}_{args.train_data}.pth"
        model_path = os.path.join(args.checkpoints_dir_to_save, model_name)
        #torch.save(model.state_dict(), model_path)

    return avg_train_loss, accuracy, highest_accuracy

#DP搜索降本函数
def downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3):
    T = np.sum(t_dw + t_dy)
    scale = Tq / T
    t_dy_q = np.floor(t_dy * scale).astype(np.int16)
    t_dw_q = np.floor(t_dw * scale).astype(np.int16)
    disco = 1.0 * np.sum(t_dy_q + t_dw_q) / Tq
    return t_dy_q, t_dw_q, disco

def main():
    args = get_args()
    setup_logging(args)
    logging.info("Training start.")
    args.device='cuda'#'cpu'
    #torch.cuda.set_device(args.gpu_id)
    
    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    model = resnet50(args, num_classes=10).to(args.device)
    model.sparse_training = args.sparse_training
    model.autocast = args.autocast
    #print(model.sparse_training)
    
    # 读取模型的层时延信息
    t_f, t_dw, t_dy = torch.load(args.layer_latency_path)
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy.numpy(), t_dw.numpy(), args.Tq)
    model.t_dy_q = np.flip(t_dy_q)#数组逆序
    model.t_dw_q = np.flip(t_dw_q)
    model.speed_up_ratio = args.speed_up_ratio * disco
    
    model.layer_importance = np.empty(len(model.param_module_names))
    
    if args.continue_training:
        # Load weights for further training
        checkpoint = torch.load(args.ckpt_dir, map_location=args.device)
        model.load_state_dict(checkpoint)
        print("Loaded weights for further training from:", args.ckpt_dir)
    
    criterion = nn.CrossEntropyLoss()
    if args.sparse_training == True:
        args.lr = args.lr * 4
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.mode=='train':
        model.train()
        
        train_data = datasets.CIFAR10(root='/taoset/Desktop/sparse_training/',
                         train=True,                         # 训练集
                         transform=transforms.ToTensor(),   
                         download=True
                        )

        test_data = datasets.CIFAR10(root='/taoset/Desktop/sparse_training/',
                        train=False,                         # 测试集
                        transform=transforms.ToTensor(),     # 转换为Tensor并归一化[0~1]
                        download=True
                        )
        
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

        highest_accuracy = 0.0  # To track the highest validation accuracy
        for epoch in range(args.train_epochs):
            train_loss, accuracy, highest_accuracy = train(train_loader, test_loader, model, criterion, optimizer, args.device, highest_accuracy, args)
            logging.info(f"Epoch {epoch+1}/{args.train_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {accuracy:.4f}")
            if accuracy == highest_accuracy:
                #print(f"Model saved with highest accuracy: {accuracy:.4f}")
                logging.info(f"Model saved with highest accuracy: {accuracy:.4f}")

        logging.info("Training completed.")
        
    elif args.mode=='test':
        model.eval()
        
        traindata_loader_function = data_loaders[args.test_data]
        test_set, test_loader= traindata_loader_function(args.test_data_dir, 'test') 
        
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                # Forward pass
                outputs = model(inputs)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total_samples += targets.size(0)
                total_correct += predicted.eq(targets).sum().item()

        test_accuracy = 100.0 * total_correct / total_samples

        print(f"Test Accuracy: {test_accuracy:.2f}%")
        logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        print("Testing completed.")
        logging.info("Testing completed.")
        

if __name__ == "__main__":
    main()
