import time
import torch
from torch import nn,optim
import d2lzh_pytorch as d2l
device = torch.device('cuda'if torch.cuda.is_available()else'cpu')
#LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),# in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),# kernel_size, stride
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
        def forward(self, img):
            feature =self.conv(img)
            output =self.fc(feature.view(img.shape[0],-1))
            return output
net =LeNet()
print(net)
#获取数据和训练模型
batch_size =256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
'''
dzl中的函数
evaluate_accuracy():
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(data_iter, net, device=None):
if device isNoneand isinstance(net, torch.nn.Module):
# 如果没指定device就使用net的device
    device = list(net.parameters())[0].device
    acc_sum, n =0.0,0
with torch.no_grad():
for X, y in data_iter:
if isinstance(net, torch.nn.Module):
    net.eval()# 评估模式, 这会关闭dropout
    acc_sum +=(net(X.to(device)).argmax(dim=1)== y.to(device)).float().sum().cpu().item()
    net.train()# 改回训练模式
else:# 自定义的模型, 3.13节之后不会用到, 不考虑GPU
if('is_training'in net.__code__.co_varnames):# 如果有is_training这个参数
# 将is_training设置成False
    acc_sum +=(net(X, is_training=False).argmax(dim=1)== y).float().sum().item()
else:
    acc_sum +=(net(X).argmax(dim=1)== y).float().sum().item()
    n += y.shape[0]
return acc_sum / n
train_ch5():
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start =0.0,0.0,0,0, time.time()
for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum +=(y_hat.argmax(dim=1)== y).sum().cpu().item()
            n += y.shape[0]
            batch_count +=1
        test_acc = evaluate_accuracy(test_iter, net)
print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
%(epoch +1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time()- start))
'''
lr, num_epochs =0.001,5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
