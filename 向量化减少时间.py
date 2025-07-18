import numpy as np
a = np.array([1,2,3,4])
print(a)

import time
a = np.random.rand(10000000)
b = np.random.rand(10000000) #维度为10000000

tic = time.time() #记录当前时间
c = np.dot(a,b)
toc = time.time()

print(c)
print("向量化维度下所用时间："+str(1000*(toc-tic))+"ms")

c = 0
tic = time.time()
for i in range(10000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("非向量化维度下所用时间："+str(1000*(toc-tic))+"ms")