import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import time
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='E:/PictureTest/fashionmnist',
                                                train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='E:/PictureTest/fashionmnist',
                                               train=False,download=True,transform=transforms.ToTensor())
# mnist_train和 mnist_test都是torch.utils.data.Datasets的子类
# 用 len()可以来获取该数据集大小
print(type(mnist_train))
print(len(mnist_train),len(mnist_test))
#通过下标访问任意一个样本
feature,label = mnist_train[0]
# Channel * Height * Width
print(feature.shape,label)
#查看训练数据集中前九个样本数据的图像内容和文本标签
X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))

#读取小批量，使用多进程来加速读取数据
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size,shuffle=True,num_workers=num_workers)
#查看读取一遍训练数据需要的时间
start = time.time()
for X,y in train_iter:
    continue
print('%2f sec' % (time.time()-start))
