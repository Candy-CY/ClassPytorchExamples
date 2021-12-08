import torch
from d2lzh import *
import numpy as np
import d2lzh_pytorch as d2l
#使用Fashion—MNIST数据集，并设置批量大小为256
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#初始化模型参数（28×28像素照片，输入向量长就是784）
#Softmax回归的权重和偏差参数分别是 784×10 和 1×10 的矩阵
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float,requires_grad=True)
b = torch.zeros(num_outputs,dtype=torch.float,requires_grad=True)
#需要模型的参数梯度
W.requires_grad_(requires_grad=True)    #注意是grad不是gard!!!!
b.requires_grad_(requires_grad=True)
#实现SOFTMAX运算
X = torch.tensor([[1,2,3],[4,5,6]])
print(X.sum(dim=0,keepdim=True)) #keepdim=True表示在结果中保留行和列两个维度
print(X.sum(dim=1,keepdim=True))
#softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上预测的概率
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp / partition   #这里应用了广播机制
X = torch.rand((2,5))
X_prob = softmax(X)
print(X_prob,X_prob.sum(dim=1))
#定义模型
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)
#为了得到标签的预测概率，我们可以使用gather函数
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y = torch.LongTensor([0,2])
y_hat.gather(1,y.view(-1,1))
#实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
#计算分类准确率:y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
print(accuracy(y_hat, y))
# 评价模型 net 在数据集data_iter上的准确率(d2l文件中也有此函数)
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
print(evaluate_accuracy(test_iter, net))
#训练模型
'''
使用小批量随机梯度下降来优化模型的损失函数。
在训练模型时，迭代周期数num_epochs和学习率lr都是可以调的超参数。
改变它们的值可能会得到分类更准确的模型。
'''
num_epochs, lr = 5, 0.1
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            for param in params:
                param.grad.data.zero_()
            l.backward() # 反向传播求梯度
            sgd(params, lr, batch_size) # 随机梯度下降算法更新参数
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
#预测：训练完成后，现在就可以演示如何对图像进行分类了
# 给定一系列图像（第三行图像输出）,比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])



