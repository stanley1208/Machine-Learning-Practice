# time complexity
import numpy as np
def fib(n):
    tmp=np.zeros(n)
    tmp[0]=1
    tmp[1]=1
    for i in range(2,n):
        tmp[i]=tmp[i-2]+tmp[i-1]

    return tmp[n-1]

# O(N)<-O(2^n)

def fib(n):
    tmp=np.zeros(n)
    tmp[0]=1
    tmp[1]=1
    for i in range(2,n):
        tmp[i]=tmp[i-2]+tmp[i-1]

    return tmp[n-1]

def fib(n):
    a,b=1,1
    c=0
    for i in range(2,n):
        c=a+b
        a=b
        b=c
    return c

# 怎麼在O(1)的時間複雜度下計算fib(n)
# 套公式
# 公式怎麼得來的?
# 提示: 轉換成矩陣連成的形式，矩陣連乘可以簡化(MATRIX DECOMPOSION)