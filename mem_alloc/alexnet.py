import time
import torch
from torch import nn, optim
import torchvision
import sys
# from test import monitor_gpu_memory
import argparse

def load_data_fashion_mnist(batch_size, resize=None, root='./data/FashionMNIST'):
    # if sys.platform.startswith('win'):
    #     num_workers = 0
    # else:
    #     num_workers = 4
    num_workers = 0 
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(args, net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    gpu_memory = []
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # print("GPU Memory Usage:")
        print(args.empty_cache)
        if args.empty_cache == 'True':
            torch.cuda.empty_cache()
        # gpu_memory.append(monitor_gpu_memory())
        print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
        gpu_memory.append(torch.cuda.memory_reserved(0)/(1024*1024))
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
            if args.empty_cache == 'True':
                torch.cuda.empty_cache()
            if batch_count % 10 == 0:
                # monitor_gpu_memory()
                print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return gpu_memory
        

def AlexNet_running(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    net = AlexNet()

    

    lr, num_epochs = 0.001, 3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    gpu_memory = train(args, net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    print(gpu_memory)
