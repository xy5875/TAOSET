#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.append('../../')


from torch.utils.data import DataLoader

import argparse
import ps

from trainer import DistTrainerParameterServer
from ps_dist.model import AlexNet

import torch
import torch.optim as optim


from torchvision import datasets, transforms



cluster_conf = {
    "ps": [
        "192.168.1.108:6001"
    ],
    "workers": [
        "192.168.1.108:6002",
        "192.168.1.107:6003"
    ]
}



def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    epoches = 16

    train_dataset = datasets.MNIST(root='./dataset/MNIST', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./dataset/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader,test_loader

# def load_test_data():



def train(worker_index,batch_size):


    train_data, test_data = load_data(batch_size)
    epoches = 10


    #model =CNN()
    model = AlexNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)


    test_trainer = DistTrainerParameterServer(train_data, test_data, model, criterion, optimizer, epoches, batch_size,eval_on_train=True,cluster_conf=cluster_conf,worker_index = worker_index)

    test_trainer.train_and_eval(train_data,test_data)

    #exporter = Exporter()
    #sig = exporter.signature('img_input', 'softmax_output')

    #saver = ms.trainer.Saver('./export/{}'.format(worker_index))
    #saver.save(model_file_name='my_model.json',
    #           weights_file_name='my_weights.npz', service_signature=sig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role',default='worker', type=str)
    parser.add_argument('--worker_index',default = '1', type=int)

    batch_size = 8

    args = parser.parse_args()

    role = args.role
    # 如果是PS角色，启动PS服务器，等待Worker连入
    if role == 'ps':
        server = ps.ParameterServiceServer(cluster_conf, sync=True)
        server.serve()
    else:
        # 如果是worker角色，则需要指定自己的index
        worker_index = args.worker_index
        train(worker_index,batch_size)
