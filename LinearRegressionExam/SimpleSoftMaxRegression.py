import torch
from torch import nn
from torch.nn import init
#from d2lzh import *
#import numpy as np
import d2lzh_pytorch as d2l

#获取和读取数据
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#定义和初始化模型
num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self,x): # x shape: (batch,1,28,28)
        y = self.linear(x.view(x.shape[0],-1))
        return y
net = LinearNet(num_inputs,num_outputs)
#形状转换功能自定义一个位于 d2l中的 FlattenLayer函数
#定义模型
from collections import OrderedDict
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten',d2l.FlattenLayer()),
        ('Linear',nn.Linear(num_inputs,num_outputs))
    ]
    )
)
#使用均值为 0，标准差为0.01的正态分布随机初始化模型的权重参数
init.normal_(net.Linear.weight,mean=0,std=0.01)
init.constant_(net.Linear.bias,val=0)
#SOFTMAX和交叉熵损失函数
loss = nn.CrossEntropyLoss()
#定义优化函数
#使用学习率为 0.1的小批量随机梯度下降作为优化算法
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
#训练模型
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)


