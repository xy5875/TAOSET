#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')

import argparse

from model import AlexNet
import torch
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from trainer import DistTrainerRingAllReduce

from torch.utils.data import Dataset, DataLoader

cluster_conf = {
    "workers": [
        "127.0.0.1:7001",
        "127.0.0.1:7002"
        # "127.0.0.1:6004"
    ]
}


# class QuadraticDataset(Dataset):
#     def __init__(self, a, b, c, num_samples):
#         super(QuadraticDataset, self).__init__()
#         self.a = a
#         self.b = b
#         self.c = c
#         self.num_samples = num_samples
#
#     def __getitem__(self, index):
#         x = torch.randn(1)  # 随机生成一个输入值
#         y = self.a * x ** 2 + self.b * x + self.c  # 计算对应的输出值
#         return x, y
#
#     def __len__(self):
#         return self.num_samples


def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    train_dataset = datasets.MNIST(root='./dataset/MNIST', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./dataset/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    # train_dataset = QuadraticDataset(a=2, b=1, c=5, num_samples=1000)
    # test_dataset = QuadraticDataset(a=2, b=1, c=5, num_samples=100)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def train(worker_index, batch_size):
    train_data, test_data = load_data(batch_size)
    epoches = 10
    batch_size = 32

    model = AlexNet()
    #model = QuadraticModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # criterion = torch.nn.MSELoss()  # 使用均方误差损失函数
    # optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

    test_trainer = DistTrainerRingAllReduce(train_data, test_data, model, criterion, optimizer, epoches, batch_size,
                                            eval_on_train=True, cluster_conf=cluster_conf, worker_index=worker_index)

    test_trainer.train_and_eval(train_data, test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_index', default='1', type=int)
    args = parser.parse_args()
    batch_size = 32

    # 如果是Worker角色，则需要指定自己的index
    worker_index = args.worker_index
    train(worker_index, batch_size)
