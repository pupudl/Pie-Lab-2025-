import numpy as np
import time

#定义成向量
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
#定义成矩阵
x3 = np.array([9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0])
x4 = np.array([9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0])

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()

x3 = x3.reshape(1,len(x3))
x4 = x4.reshape(1,len(x4))

dot = np.dot(x1,x2)        # 向量内积,只有一个元素,即multiply的每个元素的和
dot2 = np.dot(x3.T,x4)     # 矩阵乘法

toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
print ("dot2 = " + str(dot2) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###   外积得矩阵 每个元素为A[i]*B[j]
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###  A与B对应元素相乘
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")