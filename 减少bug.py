import numpy as np

a = np.random.randn(5)

print(a) #a为五个随机的高斯变量吗,eg.[ 1.72811987 -0.5105292   0.33212295  1.45270481 -0.77322423]

print(a.shape) #a的形状为(5,),其既不是行向量也不是列向量

print(a.T) #转置后与a一样

print(np.dot(a,a.T)) #与np.dot(a.T,a)得到的都是一个数

##因此建议尽量少使用(n,)形状的秩为一的数组，应尽量使用以下形式来定义
##或者使用a.reshape(5,1)来改变形状

a = np.random.randn(5,1)
print(a) #此时为一个列向量

print(a.T) #转置后为行向量

print(np.dot(a,a.T))