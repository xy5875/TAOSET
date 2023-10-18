import threading
import time


from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import abc
import ps
import time
import torch

import numpy as np

import model


class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, train_iter,test_iter,model,
                 loss_op, optimizer,
                 epoches,batchsize,
                 eval_on_train=False, *args, **kargs):

        # 计算图的输入节点，可以有多个，因此类型是list
        self.train_iter = train_iter
        self.test_iter = test_iter

        self.model = model

        # 损失函数
        self.loss_op = loss_op

        # 优化器
        self.optimizer = optimizer

        # 训练执行的epoch数
        self.epoches = epoches
        self.epoch = 0
        self.batchsize = batchsize
        # 批大小


        # 是否在训练迭代中进行评估
        self.eval_on_train = eval_on_train

        # 评估指标列表


        # self.print_iteration_interval = kargs.get(
        #     'print_iteration_interval', 100)

    def train_and_eval(self, train_iter, test_iter = None):
        '''
        开始训练(评估)流程
        '''


        # 传入数据，开始主循环
        self._variable_weights_init()
        self.main_loop(train_iter, test_iter)

    def main_loop(self, train_iter, test_iter):
        '''
        训练（评估）的主循环
        '''

        # 第一层循环，迭代epoches轮
        for self.epoch in range(self.epoches):

            # 模型训练
            self.train(train_iter)

            # 如果需要，对模型进行评估
            if self.eval_on_train and test_iter is not None:
                self.eval(test_iter)

    def train(self, train_iter):
        '''
        使用训练集进行模型训练
        '''


        # 遍历训练数据集
        for batch_idx, data in enumerate(train_iter, 0):
            inputs, target = data
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss= self.loss_op(outputs, target)
            loss.backward()
            self._optimizer_update()
            if batch_idx % 10 == 0:
                print('[%5d] loss:%.3f' % (batch_idx + 1, loss.item()))


    def eval(self,test_iter):
        '''
        使用测试集进行模型评估
        '''
        test_acc_sum, test_num = 0.0, 0
        with torch.no_grad():  # 不会求梯度、反向传播
            self.model.eval()  # 不启用 BatchNormalization 和 Dropout
            for X, y in test_iter:
                test_acc_sum += (self.model(X).argmax(dim=1) == y).float().sum().item()
                test_num += y.shape[0]
            accu = test_acc_sum / test_num
            print('test acc %.3f' % accu)



    @abc.abstractmethod
    def _variable_weights_init(self):
        '''
        权值变量初始化，具体的初始化操作由子类完成
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        '''
        调用优化器执行参数更新
        '''
        raise NotImplementedError()





class DistTrainerParameterServer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']

        ps_host = cluster_conf['ps'][0]

        self.ps_client = ps.ParameterServiceClient(ps_host)


    def _variable_weights_init(self):
        '''
        多个worker通过ps保证变量节点初始化一致
        '''
        var_weights_dict = dict()



        var_weights_dict = self.model.state_dict()

        # 把自己的初始值发送给ps，由ps决定使用哪个Worker并返回

        duplicated_var_weights_dict = self.ps_client.variable_weights_init(
            var_weights_dict)


        # 使用ps返回的初始值，重新初始化本地
        self.model.load_state_dict(duplicated_var_weights_dict)
        print('[INIT] Worker variable weights initialized')


    def _optimizer_update(self):

        # 把当前梯度push到ps上。此操作可能被block，直到所有节点都pull完成
        acc_gradient = dict()

        for n,p in self.model.named_parameters():
            acc_gradient[n] = p.grad

        #acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.batchsize)


        # 从ps把所有节点的平均梯度pull回来。此操作可能被block直到所有节点都push完成
        node_gradients_dict = self.ps_client.pull_gradients()

        for n,p in self.model.named_parameters():
            p.grad = node_gradients_dict.get(n)
        # 使用平均梯度，利用优化器的优化算法，更新本地变量
        self.optimizer.step()


if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    batch_size = 256
    epoches = 16

    train_dataset = datasets.MNIST(root='./dataset/MNIST', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./dataset/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    model = model.AlexNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    test_trainer = Trainer(train_loader,test_loader,model,criterion,optimizer,epoches, batch_size,eval_on_train=True)
    test_trainer.train_and_eval(train_loader,test_loader)
