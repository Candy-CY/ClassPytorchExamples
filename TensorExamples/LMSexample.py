"""单层感知器的例子，是简单的线性网络例子"""
#使用线性神经网络解决异或问题
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
# 载入数据(直接构建)
#构建跟多的输入属性即: 1、x1、x2、x1^2、x2^2、2*x1*x2
x_data = np.array([[1,0,0,0,0,0],
                   [1,0,1,0,0,1],
                   [1,1,0,1,0,0],
                   [1,1,1,1,1,1]])
y_data = np.array([[-1,1,1,-1]])

# 使用sklearn中的PolynomialFeatures函数生成
# 2次多项式特征生成器
poly_reg = PolynomialFeatures(degree=2)
'''
fit() ：用于从训练数据生成学习模型参数
transform()：从fit()方法生成的参数，应用于模型以生成转换数据集。
fit_transform()：在同一数据集上组合fit()和transform()的API
'''
x_poly = poly_reg.fit_transform([[0,0],
                                [0,1],
                                [1,0],
                                [1,1]])
print(x_poly)
# 标签
y_data = np.array([[-1,1,1,-1]])
# 设置权值
w = (np.random.random(6)-0.5)*2
print(w)
# 设置学习率
lr = 0.11
# 神经网络的输出
out_data = 0
# 统计迭代次数
n = 0

def update():
    global x_data,y_data,w,lr,n
    n += 1
    out_data = np.dot(x_data,w.T) # x_data是4*3矩阵，w是3*1矩阵,输出4*1矩阵
    w_c = lr*((y_data-out_data.T).dot(x_data))/int(x_data.shape[0])# 分母求得误差的总合，分子表示sample的数量，结果表示平均误差
    w = w+w_c
for i in range(1000):
    update() # 权值更新
# 正样本
x1 = [0,1]
y1 = [1,0]
# 负样本
x2 = [1,0]
y2 = [1,0]

def calculate(x,root):
    """进行公式推导，确定二次方程的解得到a/b/c返回函数的解"""
    a = w[0,5]
    b = w[0,2]+x*w[0,4]
    c = w[0,0]+x*w[0,1]+x*x*w[0,3]
    if root ==1:
        return(-b+np.sqrt(b*b-4*a*c))/(2*a)
    if root == 2:
        return(-b-np.sqrt(b*b-4*a*c))/(2*a)
# 绘图
xdata =np.linspace(-1,2) # linspace是线性划分区间，生成一系列的点
plt.figure()
plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,2),'r')
plt.plot(x1,y1,'bo') # 注意这里的plot并非是scatter
plt.plot(x2,y2,'yo')
plt.show()
