import torch
import time
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet_model import ResNet50
#from test import monitor_gpu_memory
import argparse
 
def data_ready():

#  用CIFAR-10 数据集进行实验
    batch_size = 16
    num_workers = 0
 
    cifar_train = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    cifar_test = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return cifar_train,cifar_test


def train_resnet(args, model, train_data, test_data, device):
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    def evaluate_accuracy(data_iter, model, device=None):
        if device is None and isinstance(model, torch.nn.Module):
            device = list(model.parameters())[0].device
        
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in data_iter:
                model.eval()
                acc_sum += (model(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
                model.train()
                n += y.shape[0]
        return acc_sum / n
 
    # print(iter(cifar_test).next()[0].shape)
    gpu_memory = []
    for epoch in range(10):
        # print("GPU Memory Usage:")
        print(args.empty_cache)
        if args.empty_cache == 'True':
            torch.cuda.empty_cache()
        # gpu_memory.append(monitor_gpu_memory())
        print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
        gpu_memory.append(torch.cuda.memory_reserved(0)/(1024*1024))
        model.train()
        start_time = time.time()
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for x, label in train_data:
            x, label = x.to(device), label.to(device)
 
            logits = model(x)
            loss = criteon(logits, label)
 
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += (logits.argmax(dim=1) == label).sum().item()
            n += label.shape[0]
            batch_count += 1
            if args.empty_cache == 'True':
                torch.cuda.empty_cache()
            if batch_count % 10 == 0:
                # monitor_gpu_memory()
                print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
        train_loss = train_loss_sum / batch_count
        train_acc = train_acc_sum / n
        test_acc = evaluate_accuracy(test_data, model)
        print("epoch:",epoch + 1, ', loss:', train_loss, ", train acc:", train_acc, 
                ", test acc:", test_acc, ", time:", time.time()-start_time)
    return gpu_memory
        
 

 
def ResNet50_running(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_data,test_data = data_ready()
    model = ResNet50().to(device)

    gpu_memory = train_resnet(args, model, train_data, test_data, device)

    print(gpu_memory)
