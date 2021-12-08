#二维互相关运算
import torch
from torch import nn
import d2lzh_pytorch as d2l
'''
def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y
'''
X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
K = torch.tensor([[0,1],[2,3]])
print("二维卷积计算结果：\n",d2l.corr2d(X,K))
#自定义二维卷积层：利用corr2d
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self,x):
        return d2l.corr2d(x,self.weight)+self.bias
#图像中物体的边缘检测
X = torch.ones(6,8)
X[:,2:6] = 0
print("构造后的图像数据：\n",X)
#构造一个高和宽分别为1、2的卷积核K,如何横向相邻元素相同输出为0，否则输出为非0
K = torch.tensor([[1,-1]])
Y = d2l.corr2d(X,K)
print("进行卷积操作后的结果：\n",Y)
#通过数据来学习核数组
'''
使用物体边缘检测中的输入数据X和输出数据Y来学习构造的核数组K，
首先构造一个卷积层,其卷积核被初始化为随机数组，在接下来的每一次迭代中，
都使用平方误差来比较Y和卷积层的输出，然后计算梯度来更新权重。
'''
#构造一个核数组形状是(1,2)的二维卷积层
conv2d = Conv2D(kernel_size=(1,2))
step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat-Y)**2).sum()
    l.backward()
    #梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    #梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1) % 5 == 0:
        print("step %d, loss %.3f" % (i+1,l.item()))
print("weight:\n",conv2d.weight.data)
print("bias:\n",conv2d.bias.data)
