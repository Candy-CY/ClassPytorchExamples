import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
#生成数据集
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
#features 是每行长度为2的向量，labels是每一行长度为1的向量
features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
print(features[0],labels[1])
#通过生成第二个特征feature[:,1]和标签Labels的散点图
def use_svg_display():
    #用矢量图来表示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    #设置图尺寸
    plt.rcParams['figure.figsize']=figsize

set_figsize()
plt.scatter(features[:,1].numpy(),labels.numpy(),1)
plt.show()

#读取数据：每次返回batch_size（批量大小）个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的

    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
#每个批量的特征形状为（10,2),分别对应批量大小和输入个数；标签形状为批量大小
batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break
#初始化模型参数
#将权重初始化成均值为0，标准差为0.01的正态随机数，偏差则初始化为0
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b = torch.zeros(1,dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#定义模型:线性回归的矢量计算表达式的实现
def linreg(X, w, b):
    w = w.double()
    return torch.mm(X, w) + b #mm函数做矩阵乘法
#定义损失函数
def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2
#定义优化算法
#这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的。
    # 因为一般用PyTorch计算loss时就默认已经沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
#训练模型：多次迭代模型参数
#在一个epoch中，将完整遍历一遍data_iter函数，并对训练数据集中所有样本都使用一次。
#迭代周期个数num_epochs和学习率lr都是超参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    # 训练模型一共需要num_epochs个迭代周期
    # 在每个迭代周期中，会使用训练数据集中所有样本一次(假设样本数能够被批量大小整除)
    # x,y 分别是小批量样本的特征和标签
    for X,y in data_iter(batch_size,features,labels):
        # l是有关小批量X和y的损失
        l = loss(net(X,w,b),y).sum()
        # 小批量的损失对模型参数求梯度
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w,b],lr,batch_size)
        #不要忘记梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features,w,b),labels)
    print('epoch %d,loss %f' % (epoch+1,train_l.mean().item()))
print(true_w,'\n',w)
print(true_b,'\n',b)



