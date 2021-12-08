import torch
import torch.nn as nn
import d2lzh_pytorch  as d2l

#实现含多个输入通道的互相关运算。我们只需要对每个通道做互相关运算,然后通过add_n函数来进行累加
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维(通道维)分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    print("计算前：\n",res)
    for i in range(1, X.shape[0]):  # X.shape[0]代表多少个通道,此处为2个
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    print("计算后：\n",res)
    return res
X = torch.tensor([[[0,1,2],[3,4,5],[6,7,8]],[[1,2,3], [4,5,6], [7,8,9]] ])
K = torch.tensor([[[0,1],[2,3]], [[1,2],[3,4]]])
corr2d_multi_in(X, K)
#互相关运算函数来计算多个通道的输出
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])
K = torch.tensor([[[0,1],[2,3]], [[1,2],[3,4]]])
# 构造3个卷积核
K = torch.stack([K, K+1, K+2])
print(K.shape)
#对输入数组X与核数组K做互相关运算。此时的输出含有3个通道。
# 其中第一个通道的结果与之前输入数组X与多输入通道、单输出通道核的计算结果一致。
# 输入的规模为  2 * 3 * 3 输出的规模为 3 * (3 - 2+ 1) * (3 - 2 + 1)
corr2d_multi_in_out(X, K)
# 1 * 1卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)
X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().item()  < 1e-6

