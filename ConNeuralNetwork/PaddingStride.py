#padding
import torch
from torch import nn
#定义一个函数来计算卷积层，它对输入输出做出相应的升维和降维
def comp_conv2d(conv2d,X):
    #(1,1)代表批量大小和通道数
    X = X.view((1,1)+X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) #排除不关心的前两列，批量和通道
#注意：这里是两侧分别填充一行或列，所以在两侧一共填充了两行或列
conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
X = torch.rand(8,8)
print("进行运算后的数据输出大小：\n",comp_conv2d(conv2d,X).shape)
conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(5,3),padding=(2,1))
print("进行运算后的数据输出大小：\n",comp_conv2d(conv2d,X).shape)

#Stride 步幅
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
print("进行运算后的数据输出大小：\n",comp_conv2d(conv2d,X).shape)
conv2d = nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
print("进行运算后的数据输出大小：\n",comp_conv2d(conv2d,X).shape)
