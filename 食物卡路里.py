import numpy as np

A = np.array([[56.0,0.0,4.4,68.0],
              [1.2,104.0,52.0,8.0],
              [1.8,135.0,99.0,0.9]])
print(A)

cal =A.sum(axis=0) #竖直相加  #axis=1时为水平相加
print(cal)

percentage = 100*A/cal.reshape(1,4) #reshape其实可以不用，因为cal矩阵本身就是(1,4)了
#此处对两个矩阵的+-*/操作，会使得维度不够的矩阵进行广播扩充，再进行运算

print(percentage)