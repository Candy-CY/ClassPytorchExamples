import torch
'''创建Tensor'''
#创建一个未#创建一个未初始化的5*3的张量
x=torch.empty(5,3) #每次运行的结构都不同
print(x)
#随机初始化一个5*3的张量
x=torch.rand(5,3)
print(x)
#创建一个5*3的long型全0的Tensor
x=torch.zeros(5,3,dtype=torch.long)
print(x)
#直接创建Tensor数据
x=torch.tensor([5.5,3])
print(x)
#通过现有的Tensor来进行创建
x=x.new_ones(5,3,dtype=torch.float64)
#返回tensor默认具有相同的torch.dtype
print(x)
x=torch.rand_like(x,dtype=torch.float)
#指定新数据类型
print(x)
#通过shape或者是size()来获取Tensor的形状
print(x.size())
print(x.shape)
'''Tensor的各种操作'''
'''加法操作'''
y=torch.rand(5,3)
print(torch.add(x,y))
#加法的指定输出形式
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
#加法形式——inplace
y.add_(x)
print(y)
'''索性操作'''
#索引出来的结果与原数据共享内存，既修改其中一个，另外一个会跟着修改
y=x[0,:]
print(y)
y+=1
print(y)
print(x[0,:]) #源tensor也被修改了
'''改变形状'''
#用view()来改变Tensor的形状
'''view()返回的新Tensor与源Tensor共享内存，也就是同一个Tensor
   改变其中一个数，另外一个也会跟着改变，view()仅仅是改变了对这个张量的观察角度'''
y=x.view(15)
z=x.view(-1,5) #-1所指的维度可以根据其他维度推算出来
print(x.size(),y.size(),z.size())
#改变其中一个变量值
x-=1
print(x)
print(y)
#reshape()也可以改变形状，但不能保证返回的是其拷贝
#返回不共享内存的方法是：推荐先使用clone()来创建一个副本，再使用view()
x_cp=x.clone().view(15)
x-=1
print(x)
print(x_cp)
#另一个常用的函数是item(),它可以将一个标量Tensor转换成Python number
x = torch.rand(1)
print(x)
print(x.item())
'''广播机制'''
#对两个形状不同的Tensor进行运算，可能会触发广播机制
x=torch.arange(1,3).view(1,2)
print(x)
y=torch.arange(1,4).view(3,1)
print(y)
print(x+y)
'''运算的内存开销'''
#索引、View是不会开辟新内存 但是y=y+x会开辟新内存
x=torch.tensor([1,2])
y=torch.tensor([3,4])
id_before=id(y)
y=y+x
print(id(y)==id_before) #Flase
#把x+y的结果通过[:]写进y对应的内存中
x=torch.tensor([1,2])
y=torch.tensor([3,4])
id_before=id(y)
y[:] = x+y
print(id(y)==id_before) #True
#使用out参数或者是+=也就是add()方法，也可以达到上述效果
x=torch.tensor([1,2])
y=torch.tensor([3,4])
id_before=id(y)
torch.add(x,y,out=y) # y+=x , y.add(x)
print(id(y)==id_before) # True
'''Tensor与Numpy相互转换'''
#Tensor转Numpy,Tensor和Numpy相互转换，numpy() from_numpy() 实现两者相互转换
a=torch.ones(5)
print(a)
b=a.numpy()
print(a,b)
#两个函数所产生的Tensor和Numpy中的数组共享相同的内存
a+=1
print(a,b)
b+=1
print(a,b)
#Numpy转Tenso
import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
print(a,b)
a+=1
print(a,b)
b+=1
print(a,b)
#直接用torch.tensor()将Numpy数组转换成Tensor,该方法是进行数据拷贝，返回的Tensor与源数据不会共享内存
c=torch.tensor(a)
a+=1
print(a,c)
'''Tensor ON GPU'''
"""
#以下代码只有在 Pytorch GPU版本上才会运行
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.one_like(x,device=device) #直接创建一个在GPU上的Tensor
    x = x.to(device) #等价于.to("cuda")
    z = x+y
    print(z)
    print(z.to("cpu",torch.double)) # to()还可以同时修改数据类型
"""
