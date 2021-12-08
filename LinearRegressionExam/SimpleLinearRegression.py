import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from torch import nn

#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples,num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] +true_b
labels += torch.tensor(np.random.normal(0, 0.01,size=labels.size()), dtype=torch.float)
''' 读取数据:PyTorch提供了 data 包来读取数据。
    由于data常⽤作变量名，我们将导入的data模块⽤Data代替。
    在每⼀次迭代中，我们将随机读取包含10个数据样本的⼩批量。'''
import torch.utils.data as Data
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取⼩小批量量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
#取并打印第⼀个⼩批量数据样本
for X, y in data_iter:
    print(X, y)
    break
'''定义模型'''
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        # 这里的LinearNet是随着class类写的
        super(LinearNet, self).__init__()
        # 这里已经定义了是线性网络了
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
print(net) # 使⽤print可以打印出⽹络的结
# 可以⽤ nn.Sequential 来更加⽅便地搭建网络.
# Sequential 是⼀个有序的容器,网络层将按照在传入 Sequential 的顺序依次被添加到计算图中
# 写法⼀
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传⼊其他层
    )
'''
# 写法⼆
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
         ('linear', nn.Linear(num_inputs, 1))
         # ......
         ]))
'''
print(net)
print(net[0])
#可以通过 net.parameters() 来查看模型所有的可学习参数，此函数将返回⼀个⽣成器
for param in net.parameters():
    print(param)
'''
作为⼀个单层神经网络，线性回归输出层中的神经元和输⼊层中各个输入完全连接。因此，线性回归的输出层⼜叫全连接层。
注意： torch.nn 仅支持输⼊一个batch的样本不支持单个样本输入，如果只有单个样本，可使用 input.unsqueeze(0) 来添加⼀维。
'''
#初始化模型参数
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0) # 也可以直接修改bias的data:
net[0].bias.data.fill_(0)
#定义损失函数
loss = nn.MSELoss()
#定义优化算法
'''
torch.optim 模块提供了很多常用的优化算法⽐如SGD、Adam和RMSProp等。
下面实现了创建一个⽤于优化 net 所有参数的优化器实例，并指定学习率为0.03的⼩批量随机梯度下降（SGD）为优化算法。
'''
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
'''
可以为不同⼦⽹络设置不同的学习率，这在finetune时经常用到，举例如下：
optimizer =optim.SGD([
# 如果对某个参数不指定学习率，就使用最外层的默认学习率
{'params': net.subnet1.parameters()}, # lr=0.03
{'params': net.subnet2.parameters(), 'lr': 0.01}
], lr=0.03)
'''
#修改 optimizer.param_groups 中对应的学习率
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
#训练模型：在使⽤ Gluon训练模型时，我们通过调⽤ optim 实例的 step 函数来迭代模型参数。
# 按照⼩批量随机梯度下降的定义，我们在 step 函数中指明批量⼤小，从⽽对批量中样本梯度求平均。
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward() # 反向传播
        optimizer.step() # 这一步是对模型参数的优化
    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
