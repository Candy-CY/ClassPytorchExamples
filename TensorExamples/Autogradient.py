import torch
'''自动求梯度'''
#Tensor
#梯度计算
x=torch.ones(2,2,requires_grad=True)
print(x)
# x是直接创建，所以没有grad_fn
print(x.grad_fn)
#进行一下运算操作
y=x+2
print(y)
print(y.grad_fn)
#判断是否为叶子节点
print(x.is_leaf,y.is_leaf)
#进行一些复杂的运算操作
z=y*y*3
out=z.mean()
print(z,out)
#通过.requires_grad_()来用in-place的方式改变requires_grad属性
a=torch.randn(2,2)
#缺失情况下，默认 requires_grad=Flase
a=((a*3)/(a-1))
print(a.requires_grad) # Flase
a.requires_grad_(True)
print(a.requires_grad) # True
b=(a*a).sum()
print(b.grad_fn)
'''梯度'''
#out是一个标量，所以要调用backward()时不需要指定求导变量
out.backward() #等价于out.backward(torch.tensor(1.))
print(x)
print(x.grad)
#反向传播一次，grad是累加的
out2=x.sum()
print(out2)
out2.backward()
print(x.grad)
#在grad更新时，每一次运算后都需要将上一次的梯度记录清空
out3=x.sum()
x.grad.data.zero_()
print(out3)
out3.backward()
print(x.grad)
#实际例子
x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = 2*x
z = y.view(2,2)
print(z)
#现在的y并不是一个标量。所以在调用backward时需要传入一个与y同形的权重向量进行加权求和得到一个标量
v = torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
z.backward(v)
print(x.grad) # x.grad 与 x 同形张量
'''中断梯度的例子'''
x = torch.tensor(1.0,requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1+y2
print(x.requires_grad)
print(y1,y1.requires_grad) # True
print(y2,y2.requires_grad) # Flase
print(y3,y3.requires_grad) # True
# y3 对 x求梯度
y3.backward()
#有关于y2的梯度是不会回传的
print(x.grad)
'''如果我们想要修改tensor的数值，但是又不想影响反向传播'''
#进行如下操作
x = torch.ones(1,requires_grad=True)
print(x.data) #还是一个Tensor
print(x.data.requires_grad)  #已经独立于计算图外
y = 2 * x
x.data *=100 #只是改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x) #更改data的值也会影响Tensor的值
print(x.grad)
