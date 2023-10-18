# -*- coding: utf-8 -*-


import sys

import numpy as np

from model import AlexNet
from ps_dist.proto import common_pb2
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim




class DistCommon(object):
    @staticmethod
    def _serialize_proto_node_gradients(grad,batch_size=32):
        '''
        把'节点-梯度'dict序列化成protobuf对象
        '''

        dict_entries = []
        for n, p in grad.items():
            key = str(n)
            array = p

            dim = np.array(array.shape).flatten()

            array = np.array(array.cpu()).flatten()

            entry = common_pb2.ArrayRequest(key=key, array=array, dim=dim)
            dict_entries.append(entry)
        #request = common_pb2.ReqDictionary(entry=dict_entries,acc_no = batch_size)

        return dict_entries,batch_size

    @staticmethod
    def _deserialize_proto_node_gradients(request):
        '''
        把protobuf对象，反序列化为'节点-梯度'dict
        '''
        request = request
        dict_obj = {}
        dim_list = {}
        #acc_no = request.acc_no
        acc_no =1
        for entry in request.entry:
            dict_obj[entry.key] = entry.array
            dim_list[entry.key] = entry.dim

        grad = {}
        for item in dict_obj.items():
            key = item[0]
            value = np.reshape(dict_obj[key], dim_list[key])
            value = torch.tensor(value, dtype=torch.float32)
            grad[key] = value
        return grad,acc_no

    @staticmethod
    def _serialize_proto_variable_weights(model_para):
        '''
        把'变量-权值'dict序列化成protobuf对象
        '''
        dict_entries = []
        #model_para = model.state_dict()
        for key, value in model_para.items():
            rdim = np.array(value.shape).flatten()
            rarray = np.array(value.cpu()).flatten()
            rkey = str(key)
            rentry = common_pb2.ArrayResponse(key=rkey, array=rarray, dim=rdim)
            dict_entries.append(rentry)
        responce = common_pb2.ResDictionary(entry=dict_entries)

        return responce

    @staticmethod
    def _deserialize_proto_variable_weights(responce):
        '''
        把protobuf对象，反序列化为'变量-权重'dict
        '''
        dict_obj = {}
        dim_list = {}
        para ={}
        responce = responce
        for entry in responce.entry:
            dict_obj[entry.key] = entry.array
            dim_list[entry.key] = entry.dim
        for item in dict_obj.items():
            key = item[0]
            value = np.reshape(dict_obj[key], dim_list[key])
            value = torch.tensor(value, dtype=torch.float32)
            para[key] = value

        return para

if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    batch_size = 256
    epoches = 16

    train_dataset = datasets.MNIST(root='./dataset/MNIST', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./dataset/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    model = AlexNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    grad_true = {}
    for i in range(epoches):
        for batch_idx,data in enumerate(train_loader,0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            for n ,p in model.named_parameters():
                grad_true[n] = p.grad


            communication  =DistCommon()
            request = DistCommon._serialize_proto_node_gradients(grad_true,batch_size)

            #print(request)
            grad,acc_no = DistCommon._deserialize_proto_node_gradients(request)
            #print(acc_no)
            for key,value in grad.items():
                if key=='layer1.0.weight':

                    value2 = grad_true.get(key)
                   # print(value)
                   # print(value2)

