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
import all_reduce as allreduce

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
        for n, p in self.model.named_parameters():
            if n == 'fc4.bias':
                print("model_old_para", p)
        self._variable_weights_init()
        for n, p in self.model.named_parameters():
            if n == 'fc4.bias':
                print("model_new_para", p)
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
            for n, p in self.model.named_parameters():
                if n == 'fc4.bias':
                    print("model_old_para",p)
                    print("model_self_grad", p.grad)
            if batch_idx % 10 == 0:
                print('[%5d] loss:%.3f' % (batch_idx + 1, loss.item()))

            self._optimizer_update()
            for n, p in self.model.named_parameters():
                if n == 'fc4.bias':
                    print("model_new_para", p)
                    print("model_new_grad", p.grad)

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



class DistTrainerRingAllReduce(Trainer):
    '''
    Ring All-Reduce模式的分布式训练
    '''

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

        # 读取集群配置信息和自身信息
        self.cluster_conf = kargs['cluster_conf']
        self.worker_index = kargs['worker_index']

        self.workers = self.cluster_conf['workers']
        self.worker_num = len(self.workers)
        self.host = self.workers[self.worker_index]

        self.step = self.worker_num-1

        # 根据集群的环状拓扑结构确定右邻居
        self.target_host = self.workers[(
            self.worker_index + 1) % self.worker_num]

        # 本节点是否已被初始化
        self.is_init = False
        self.init_cond = threading.Condition()

        self.cur_partion_index = self.worker_index
        self.partition = []

        self.acc_no = 1

        # 获取所有可训练节点
        #self.variables = get_trainable_variables_from_graph()
        self.model_state=dict()
        self.model_variables = self.model.state_dict()

        self.variables = list(self.model_variables.keys())


        # 根据worker的总数量，对即将更新的变量节点列表进行等长切分
        self._partition_variables()

        # 用于控制梯度的发送和接收
        self.is_recieved = False
        self.recieved_gradients = None
        self.recieved_acc_no = None
        self.cond = threading.Condition()

        # 创建本节点的梯度接收服务
        allreduce.RingAllReduceServer(
            self.host, self.worker_index,
            self._variable_weights_init_callback,
            self._scatter_callback,
            self._gather_callback).serve()

        # 创建连接目标节点的梯度发送client
        self.client = allreduce.RingAllReduceClient(self.target_host)


    def _variable_weights_init(self):


        var_weights_dict = self.model.state_dict()
        print('[INIT] Send variable init weights to worker ', self.target_host)

        # 第一个节点不需要等待，使用默认值更新给下一个节点
        if self.worker_index == 0:
            self.client.variable_weights_init(var_weights_dict)
        else:
            self.init_cond.acquire()
            while not self.is_init:
                self.init_cond.wait()
            self.init_cond.release()
            self.client.variable_weights_init(var_weights_dict)


    def _variable_weights_init_callback(self, var_weights_dict):

        # 第一个节点不需要接收上一个节点的初始值
        if self.worker_index != 0:
            print('[INIT] Variables initializing weights from last worker node...')
            # for var_name, weights in var_weights_dict.items():
            #     update_node_value_in_graph(var_name, weights)
            self.model.load_state_dict(var_weights_dict)



        # 已初始化完成，通知发送流程
        self.init_cond.acquire()
        self.is_init = True
        self.init_cond.notify_all()
        self.init_cond.release()



    def _optimizer_update(self):

        # 共执行 N-1 次scatter操作，把本worker的梯度切片发送给下一个worker
        # 同时接收左邻居发送过来的梯度，累加到自己的对应切片上
        self.acc_no = self.batchsize

        # for n, p in self.model.named_parameters():
        #     print(n)
        #     #print("model_old_para", p)
        #     print("model_old_grad",p.grad)


        for scatter_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            cur_acc_no = self.acc_no if scatter_index == 0 else self.recieved_acc_no


            # 把自身的一个数据分块发送给右邻居
            self.client.send(gradients_part, cur_acc_no, 'scatter')

            # 等待接收并处理完左邻居节点的数据
            self._wait_for_recieve('scatter')

        # 然后执行 N-1 次all-gather操作，把本worker的梯度切片发送给下一个worker
        # 同时接收上一个worker发送过来的梯度并替换自己的对应切片

        for gather_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            self.client.send(gradients_part, 0, 'gather')
            self._wait_for_recieve('gather')

        # for n, p in self.model.named_parameters():
        #     print(n)
        #     #print("model_new_para", p)
        #     print("model_new_grad",p.grad)

        self.optimizer.step()


    def _partition_variables(self):
        '''
        根据worker的总数量，对即将更新的权值变量列表进行等长切分
        '''


        var_num = len(self.variables)
        #print(var_num)
        part_length = int(var_num / self.worker_num)
        assert part_length > 0

        start = 0
        end = start + part_length
        for i in range(self.worker_num - 1):
            self.partition.append((start, end))
            start = end
            end = start + part_length

        self.partition.append((start, var_num))


    def _get_gradients_partition(self):
        '''
        获取下一个梯度切片
        '''
        start, end = self.partition[self.cur_partion_index]

        part_variables = self.variables[start:end]
        self.cur_partion_index = (
            self.cur_partion_index + self.step) % self.worker_num
        part_gradients = dict()

        for var in part_variables:
            for n,p in self.model.named_parameters():
                if var == n:
                    part_gradients[var] = p.grad

        #print("part",part_gradients)
        return part_gradients


    def _scatter_callback(self, node_gradients_dict, acc_no):
        '''
        Scatter 阶段的回调函数，接收上一个worker发送过来的梯度和样本数
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            # 把接收到的梯度缓存下来
            self.recieved_gradients = node_gradients_dict
            #print("node_gradients_dict",node_gradients_dict)
            #print("self.recieved_gradients",self.recieved_gradients)

            self.recieved_acc_no = acc_no
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _gather_callback(self, node_gradients_dict):
        '''
        All-gather 阶段的回调函数，接收上一个worker发送来的梯度
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            self.recieved_gradients = node_gradients_dict
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _wait_for_recieve(self, stage):
        '''
        等待梯度，并把接收到的梯度更新到优化器中
        '''
        if self.cond.acquire():
            while not self.is_recieved:
                self.cond.wait()
            #print("stage",stage)
            #print("self.recieved_gradients",self.recieved_gradients)
            # 如果是scatter阶段则累加梯度，同时累加样本数
            if stage == 'scatter':
                self.apply_gradients(
                    self.recieved_gradients,  summarize=True, acc_no=self.worker_num)

            # 如果是all-gather阶段则覆盖梯度，样本数保持不变
            else:
                self.apply_gradients(
                    self.recieved_gradients, summarize=False, acc_no=self.worker_num)

            self.is_recieved = False

            # 梯度已被更新，通知接收流程继续接收新的梯度
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()

    def apply_gradients(self,node_gradients_dict,summarize=False,acc_no=None):
        for n,p in self.model.named_parameters():
            self.model_state[n] = p.grad
        #print("model_grad_old",self.model_state)
        #print("node_gradients_dict",node_gradients_dict)
        for name,pred in node_gradients_dict.items():
            if summarize:
                self.model_state[name] += pred
            else:
                self.model_state[name] = pred

        for n,p in self.model.named_parameters():
            new_grad = self.model_state[n]
            p.grad = new_grad

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                # 传入的是平均梯度, 强制让acc_no变为1，避免梯度更新时重复平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no



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
