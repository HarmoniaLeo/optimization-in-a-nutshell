import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库
from scipy import linalg

n=4	#x的长度

def myFunc(x):
    return  #目标方程

x=np.zeros(n)	#初值点
rho=0.6
beta=1
e=0.001
sigma=0.4
k=0
tar=Function(myFunc)
while tar.norm(x)>e:
    try:
        d=linalg.solve(tar.hesse(x),-tar.grad(x))
        if tar.value(x)-tar.value(x+d)<0:
            d=-tar.grad(x)
    except Exception:
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
