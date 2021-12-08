#定义两个1000维度的向量
import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)
#向量相加的方法就是将两个向量按元素逐一进行标量相加
start=time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time()-start)
#另一种直接相加的方式
start = time()
d = b+a  #后者比前者更加省时
print(time()-start)
#部分矢量计算运用到了广播机制
a = torch.ones(3)
b = 10
print(a+b)

