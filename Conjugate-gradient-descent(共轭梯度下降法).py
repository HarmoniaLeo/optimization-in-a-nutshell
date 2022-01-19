import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库
from scipy import linalg

n=2	#x的长度

def myFunc(x):  #x是一个包含所有参数的列表
    return x[0]**2 + 2*x[1]**2 + 2*x[0] - 6*x[1] +1 #目标函数

x=np.zeros(n)	#初始值点
e=0.001
beta1=1
sigma=0.4
rho=0.55
tar=Function(myFunc)
k=0
d=-tar.grad(x)
while tar.norm(x)>e:
    a=1
    if not (tar.value(x+a*d)<=tar.value(x)+rho*a*dot(turn(tar.grad(x)),d) and \
        np.abs(dot(turn(tar.grad(x+a*d)),d))>=sigma*dot(turn(tar.grad(x)),d)):
        a=beta1
        while tar.value(x+a*d)>tar.value(x)+rho*a*dot(turn(tar.grad(x)),d):
            a*=rho
        while np.abs(dot(turn(tar.grad(x+a*d)),d))<sigma*dot(turn(tar.grad(x)),d):
            a1=a/rho
            da=a1-a
            while tar.value(x+(a+da)*d)>tar.value(x)+rho*(a+da)*dot(turn(tar.grad(x)),d):
                da*=rho
            a+=da
    lx=x
    x=x+a*d
    beta=np.max((dot(turn(tar.grad(x)),tar.grad(x)-tar.grad(lx))/(tar.norm(lx)**2),0))	#PRP+
    beta=(tar.norm(x)**2)/(tar.norm(lx)**2)	#FR
    d=-tar.grad(x)+beta*d
    k+=1
    print(k)
print(x)