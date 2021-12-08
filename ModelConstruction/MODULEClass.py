import torch
from torch import nn
from collections import OrderedDict
#通过继承module类的方式来构造模型
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
X = torch.rand(2, 784)
net = MLP()
print("输出MLP网络：\n",net)
#该函数会调用MLP类定义的forward函数来完成前向计算
print("输出前向计算结果：\n",net(X))
#通过Sequential类定义简单的模型
class Mysequential(nn.Module):
    def __init__(self, *args):
        super(Mysequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input
#通过forward函数就可以看出Sequential类实现的是简单的串联各层
net = Mysequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        )
print("输出Mysequential网络：\n",net)
print("输出前向计算结果：\n",net(X))
# 使用ModuleList类
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print("输出使用ModuleList的网络：\n",net)
#使用ModuleDict类
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print("输出网络的output层：\n",net.output)
print("输出使用ModuleDict的网络：\n",net)
#构建复杂模型
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)
    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()
X = torch.rand(2, 20)
net = FancyMLP()
print("输出FancyMLP网络：\n",net)
print("输出前向计算结果：\n",net(X))
