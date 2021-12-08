import numpy as np
import matplotlib.pyplot as plt
# 输入数据
X = np.array([[1, 3, 3],
              [1, 4, 3],
              [1, 1, 1]])
# 存为标签（一一对应数据，（3，3）（4，3）是1 （1，1）是-1）
Y = np.array([1, 1, -1])
# 随机权值，三行一列，取值范围（-1，1）
W = (np.random.random(3) - 0.5) * 2
print('W=', W)
# 设置学习率
lr = 0.10
# 设置迭代次数
n = 0
# 设置输出值
O = 0
def update():
    global X, Y, W, lr, n
    n += 1
    O = np.sign(np.dot(X, W.T))
    W_C = lr * ((Y - O.T).dot(X)) / X.shape[0]    # 平均权值
    W = W_C + W           # 修改权值

for _ in range(100):
    update()      # 更新权值
    print(W)
    print(n)
    O = np.sign(np.dot(X, W.T))    # 计算神经网络输出
    if (O == Y.T).all():          # 如果实际输出等于期望输出，模型收敛
        print("finished")
        print("epoch:", n)
        break

# 正样本（标签为1）
X1 = [3, 4]
Y1 = [3, 3]

# 负样本（标签为0）
X2 = [1]
Y2 = [1]

# 计算斜率
k = -W[1] / W[2]
# 计算截距
d = -W[0] / W[2]
print('k=', k)
print('d=', d)

# 作图
xdata = np.linspace(0, 5)
plt.figure()
plt.plot(xdata, xdata * k + d, 'r')
plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'yo')
plt.show()
