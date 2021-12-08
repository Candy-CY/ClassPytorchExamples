import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import d2lzh_pytorch as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
train_data = pd.read_csv('E:\\PythonCVCourse\\Kaggle-House-Price-Forecast-main\\dataset\\train.csv')
test_data = pd.read_csv('E:\\PythonCVCourse\\Kaggle-House-Price-Forecast-main\\dataset\\test.csv')
#输出训练数据集和测试数据集的大小
print("训练数据集大小：",train_data.shape)
print("测试数据集大小：",test_data.shape)
#查看前四个样本的前四个特征和后两个特征和标签
print("前四个样本的前四个特征和后两个特征和标签:")
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
#将训练数据集和测试数据集的79个特征按样本连结
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
#预处理数据 standardization
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / (x.std())
)
#标准化后，每一个特征的均值都变成了0，所以可以用0来替代缺失值
all_features = all_features.fillna(0)
#将离散数据转换成指示特征
all_features = pd.get_dummies(all_features,dummy_na=True)
print("all_features.shape:",all_features.shape)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values,dtype=torch.float).view(-1,1)
#训练模型：使用基本的线性回归模型和平方损失函数
loss = torch.nn.MSELoss()
def get_net(feature_num):
    net = nn.Linear(feature_num,1)
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)
    return net
#对数均方根误差的实现
def log_rmse(net,features,labels):
    with torch.no_grad():
        # 将小于1的值设置为1，使得取对数时数值更加稳定
        clipped_preds = torch.max(net(features),torch.tensor(0.1))
        rmse = torch.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
        return rmse.item()
#使用了Adam优化算法
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
#k折交叉验证
'''
k 折交叉验证
由于验证数据集不参与模型训练，当训练数据不够用时，预留大量的验证数据显得太奢侈。一种改善的方法是k折交叉验证
（k-fold cross-validation）。在k折交叉验证中，我们把原始训练数据集分割成k个不重合的子数据集，
然后我们做k次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他k−1个子数据集来训练模型。
在这k次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这k次训练误差和验证误差分别求平均。
'''
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid
#在 k折交叉验证中我们训练 k次并返回训练和验证的平均误差。
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'% (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
#模型选择：使用一组未经调优的超参数并计算交叉验证误差。可以改动这些超参数来尽可能减小平均测试误差。
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'% (k, train_l, valid_l))
#预测并在Kaggle提交结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('E:\\PythonCVCourse\\Kaggle-House-Price-Forecast-main\\dataset\\submission.csv', index=False)
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
