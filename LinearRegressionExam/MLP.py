import torch
import numpy as np
import d2lzh_pytorch as d2l
#手动实现一个多层感知机
#继续使用Fashion—MNIST数据集，使用多层感知机来对图像进行分类
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
'''
定义模型参数：Fashion—MNIST中的图像形状为 28×28，类别数是10. 本节中依然采用28×28=784的向量
因此输入个数为784，输出个数为10，设置超参数隐藏单元个数为256
'''
num_inputs, num_outputs, num_hiddens = 784,10,256
#np.random.normal(loc= , scale= , size= ) 接收三个参数，normal在这里是指正态分布.
#该方法通过loc和scale指定正态分布的均值和方差，返回一个数组，内容是从这个分布中随机取得的值，而size就是指定这个数组的大小。
W1 = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)
W2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
b2 = torch.zeros(num_outputs,dtype=torch.float)
#输入层作为隐藏层的输入，隐藏层的输出作为输出层的输入
params = [W1,b1,W2,b2]
for param in params:
    param.requires_grad_(requires_grad = True)
#定义激活函数 激活函数一般有三种：ReLU()、Sigmoid()、tanh()
def relu(X): # ReLU函数：(max,0)
    return torch.max(input=X,other= torch.tensor(0.0))
#定义模型
def nex(X):
    X = X.view((-1,num_inputs))
    H = relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H,W2)+b2
#定义损失函数
loss = torch.nn.CrossEntropyLoss()
#训练模型
'''
原书的mxnet中的SoftmaxCrossEntropyLoss在反向传播的时候相对于沿batch维求和，而PyTorch默认的是平均。
所以PyTorch计算出来的loss要比mxnet小很多（大概是1/batch_size的量级），由此可知：反向传播的梯度也小很多。
'''
num_epochs, lr = 5,100
d2l.train_ch3(nex,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)



