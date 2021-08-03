import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库
from scipy import linalg

n=3 #x的长度
mu=2 #μ的初值

def func(x):
    return #函数

def hj(x):
	#构造数组h，第j位是第j+1个等式限制条件计算的值
    return h

def gi(x):
    #构造数组g，第i位是第i+1个等式限制条件计算的值
    return g

def S(x):
    h=hj(x)
    g=gi(x)
    return np.sum(np.power(h,2))+np.sum(np.power(np.where(g<0,g,0),2))

def myFunc(x):
    return  func(x)+S(x)*mu*0.5

sigma2=1.5	#放大因子
e2=0.001
x=np.array([2.0,2.0,2.0])	#初值点
k1=0
while mu*S(x)>=e2:
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
        d=-tar.grad(x)+beta*d
        k+=1
        print(k1,k)
    mu*=sigma2
    k1+=1
print(x)
