import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库

n=2	#x的长度

def myFunc(x):  #x是一个包含所有参数的列表
    return x[0]**2 + 2*x[1]**2 + 2*x[0] - 6*x[1] +1 #目标函数

x=np.zeros(n)	#初值点
rho=0.6
beta=1
sigma=0.4
e=0.001
k=0
tar=Function(myFunc)
while tar.norm(x)>e:
    d=-tar.grad(x)
    a=1
    if not (tar.value(x+a*d)<=tar.value(x)+rho*a*dot(turn(tar.grad(x)),d) and dot(turn(tar.grad(x+a*d)),d)>=sigma*dot(turn(tar.grad(x)),d)):
        a=beta
        while tar.value(x+a*d)>tar.value(x)+rho*a*dot(turn(tar.grad(x)),d):
            a*=rho
        while dot(turn(tar.grad(x+a*d)),d)<sigma*dot(turn(tar.grad(x)),d):
            a1=a/rho
            da=a1-a
            while tar.value(x+(a+da)*d)>tar.value(x)+rho*(a+da)*dot(turn(tar.grad(x)),d):
                da*=rho
            a+=da
    x+=a*d
    k+=1
    print(k)
print(x)
