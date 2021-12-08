#多项式函数拟合实验
import numpy as np
import d2lzh_pytorch as d2l
import torch
'''
三阶多项式：y=1.2x-3.4x²+5.6x³+5+Π
噪声项Π服从均值为0，标准差为0.01的正态分布，训练集和测试集的样本数量都设为100
'''
n_train,n_test,true_w,true_b = 100,100,[1.2,-3.4,5.6],5
features = torch.rand((n_train+n_test,1))
#torch.pow(),对输入的每分量求幂次运算,1是指维数按横着拼
poly_features = torch.cat((features,torch.pow(features,1),torch.pow(features,3)),1)
labels = (true_w[0]*poly_features[:,0]+true_w[1]*poly_features[:,1]+
          true_w[2]*poly_features[:,2]+true_b)
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
#查看生成数据集的前两个样本
print("查看前两个样本：\n",features[:2],poly_features[:2],features[:2])
#定义、训练和测试模型
'''
作图函数：semilogy(),其中对y轴使用了对数尺度
'''
# 本函数已保存在d2lzh包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()
num_epochs,loss = 100,torch.nn.MSELoss()
def fit_and_plot(train_features, test_features, train_labels,test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear⽂档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了
    batch_size = min(10, train_labels.shape[0])
    # 定义数据集
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    # 训练数据打乱
    train_iter = torch.utils.data.DataLoader(dataset, batch_size,shuffle=True)
    # 优化器 随机梯度下降SGD
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 损失函数 这里通过 view 函数将每张原始图像改成⻓度为 num_inputs 的向量。
            l = loss(net(X), y.view(-1, 1))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features),train_labels).item())
        test_ls.append(loss(net(test_features),test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss',test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,'\nbias:', net.bias.data)
#三阶多项式函数拟合（正常）
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],labels[:n_train], labels[n_train:])
#线性函数拟合（欠拟合）
fit_and_plot(features[:n_train, :], features[n_train:, :],labels[:n_train],labels[n_train:])
#训练样本不足（过拟合）
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :],labels[0:2],labels[n_train:])

