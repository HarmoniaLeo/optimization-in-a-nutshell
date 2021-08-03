import numpy as np
from Function import Function	#定义法求导工具
from lagb import *	#线性代数工具库
from scipy import linalg

n=4 #x的长度

def myFunc(x):
    return  #目标方程

k=0
x=np.zeros(n)	#初始值点
e=0.001
sigma=0.4
rho=0.55
beta=1
tar=Function(myFunc)
H=np.array(np.eye(n))
while tar.norm(x)>e:
    a=1
    d=-dot(H,tar.grad(x))
    if not (tar.value(x+a*d)<=tar.value(x)+rho*a*dot(turn(tar.grad(x)),d) and \
        dot(turn(tar.grad(x+a*d)),d)>=sigma*dot(turn(tar.grad(x)),d)):
        a=beta
        while tar.value(x+a*d)>tar.value(x)+rho*a*dot(turn(tar.grad(x)),d):
            a*=rho
        while dot(turn(tar.grad(x+a*d)),d)<sigma*dot(turn(tar.grad(x)),d):
            a1=a/rho
            da=a1-a
            while tar.value(x+(a+da)*d)>tar.value(x)+rho*(a+da)*dot(turn(tar.grad(x)),d):
                da*=rho
            a+=da
    x1=x+a*d
    if tar.norm(x1)<e:
        print(k+1)
        x=x1
        break
    s=x-x1
    y=tar.grad(x)-tar.grad(x1)
    H=H-muldot(H,y,turn(y),H)/muldot(turn(y),H,y)+dot(s,turn(s))/dot(turn(s),y)
    x=x1
    k+=1
    print(k)
