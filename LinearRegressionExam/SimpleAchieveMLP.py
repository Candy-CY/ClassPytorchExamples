#使用Gluon来实现多层感知机
import torch
import numpy as np
from torch import nn
from torch.nn import init
import d2lzh_pytorch as d2l
#定义模型
'''与softmax唯一不同在于，多加了一个全连接层作为隐藏层，它的隐藏单元个数为256，并且使用了ReLU函数作为激活函数'''
num_inputs, num_outputs, num_hiddens = 784,10,256
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs),
)
for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)
#读取数据并训练模型
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)

