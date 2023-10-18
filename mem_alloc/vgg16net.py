import torch
# from mem_swap import monitor
# from mem_swap import tensor
from torch import nn,optim,tensor
from torch.utils.data import DataLoader
from torchvision.utils import  make_grid
from torchvision import datasets,transforms
import numpy as np
from matplotlib import pyplot as plt
import time
#from test import monitor_gpu_memory
from Vgg16_model import *
import argparse

mem_recycle = False

def set_mem_recycle(value):
    global mem_recycle
    mem_recycle = bool(value)


#数据获取(数据增强,归一化)
def transforms_RandomHorizontalFlip():


    #Normalize()是归一化过程,ToTensor()的作用是将图像数据转换为(0,1)之间的张量,Normalize()则使用公式(x-mean)/std
    #将每个元素分布到(-1,1). 归一化后数据转为标准格式,
    transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.485,0.456,0.406),(0.226,0.224,0.225))])


    #root:cifar-10 的根目录,data_path
    #train:True=训练集, False=测试集
    #transform:(可调用,可选)-接收PIL图像并返回转换版本的函数
    #download:true=从互联网上下载数据,并将其放在root目录下,如果数据集已经下载,就什么都不干
    train_dataset=datasets.CIFAR10(root='./data/CIFAR10',train=True,transform=transform_train,download=True)
    # test_dataset=datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)
    # return train_dataset
    return train_dataset


def data_ready():
    #数据增强:随机翻转
    train_dataset=transforms_RandomHorizontalFlip()
    # train_dataset=transforms_RandomHorizontalFlip()


    '''
    #Dataloader(....)
    dataset:就是pytorch已有的数据读取接口,或者自定义的数据接口的输出,该输出要么是torch.utils.data.Dataset类的对象,要么是继承自torch.utils.data.Dataset类的自定义类的对象

    batch_size:如果有50000张训练集,则相当于把训练集平均分成(50000/batch_size)份,每份batch_size张图片
    train_loader中的每个元素相当于一个分组,一个组中batch_size图片,

    shuffle:设置为True时会在每个epoch重新打乱数据(默认:False),一般在训练数据中会采用
    num_workers:这个参数必须>=0,0的话表示数据导入在主进程中进行,其他大于0的数表示通过多个进程来导入数据,可以加快数据导入速度
    drop_last:设定为True如果数据集大小不能被批量大小整除的时候,将丢到最后一个不完整的batch(默认为False)
    '''

    num_workers = 0
    batch_size=32   #每次喂入的数据量

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    # test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return train_loader

def train_vgg16(args, model, train_loader, device):

    num_print=100
    epoch_num=5 #总迭代次数
    lr=0.01
    step_size=10  #每n次epoch更新一次学习率

    #在多分类情况下一般使用交叉熵
    criterion=nn.CrossEntropyLoss()
    '''
    params(iterable)-待优化参数的iterable或者定义了参数组的dict
    lr(float):学习率

    momentum(float)-动量因子

    weight_decay(float):权重衰减,使用的目的是防止过拟合.在损失函数中,weight decay是放在正则项前面的一个系数,正则项一般指示模型的复杂度
    所以weight decay的作用是调节模型复杂度对损失函数的影响,若weight decay很大,则复杂的模型损失函数的值也就大.

    dampening:动量的有抑制因子


    optimizer.param_group:是长度为2的list,其中的元素是两个字典
    optimzer.param_group:长度为6的字典,包括['amsgrad','params','lr','weight_decay',eps']
    optimzer.param_group:表示优化器状态的一个字典

    '''
    optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.8,weight_decay=0.001)

    '''
    scheduler 就是为了调整学习率设置的，我这里设置的gamma衰减率为0.5，step_size为10，也就是每10个epoch将学习率衰减至原来的0.5倍。

    optimizer(Optimizer):要更改学习率的优化器
    milestones(list):递增的list,存放要更新的lr的epoch
    gamma:(float):更新lr的乘法因子
    last_epoch:：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1
    '''

    schedule=optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)



    #训练
    loss_list=[]  #为了后续画出损失图
    start=time.time()

    gpu_memory = []
    #train
    for epoch in range(epoch_num):
        # print("GPU Memory Usage:")
        print(args.empty_cache)
        if args.empty_cache == 'True':
            torch.cuda.empty_cache()
        # gpu_memory.append(monitor_gpu_memory())
        print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
        gpu_memory.append(torch.cuda.memory_reserved(0)/(1024*1024))

        ww = 0
        running_loss=0.0
        #0是对i的给值(循环次数从0开始技术还是从1开始计数的问题):
        #???
        for i,(inputs,labels) in enumerate(train_loader,0):


            #将数据从train_loader中读出来,一次读取的样本是32个
            inputs,labels=inputs.to(device),labels.to(device)
            #用于梯度清零,在每次应用新的梯度时,要把原来的梯度清零,否则梯度会累加
            optimizer.zero_grad()
            outputs = model(inputs)


            loss=criterion(outputs,labels).to(device)

            #反向传播,pytorch会自动计算反向传播的值
            loss.backward()
            #对反向传播以后对目标函数进行优化
            optimizer.step()


            running_loss+=loss.item()
            loss_list.append(loss.item())

            if(i+1)%num_print==0:
                print('[%d epoch,%d]  loss:%.6f' %(epoch+1,i+1,running_loss/num_print))
                running_loss=0.0
                if args.empty_cache == 'True':
                    torch.cuda.empty_cache()
                print("GPU Memory Usage:", torch.cuda.memory_reserved(0)/(1024*1024), 'MB')
            
        lr_1=optimizer.param_groups[0]['lr']
        # print("learn_rate:%.15f"%lr_1)
        print("%d epoch completed, learn_rate:%.15f"%(epoch+1, lr_1))
        schedule.step()

    end=time.time()
    print("time:{}".format(end-start))
    return gpu_memory



def Vgg16Net_running(args):

    # prealloc(0.01)

    #模型,优化器
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_loader = data_ready()
    model = Vgg16_net().to(device)

    gpu_memory = train_vgg16(args, model,train_loader, device)

    print(gpu_memory)

