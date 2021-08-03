from Function import Function	#定义法求导工具
import numpy as np
from scipy import linalg
from lagb import *	#线性代数工具库

n=4	#待定系数数
y=np.array([])
t=np.array([])
#样本列表

def myFunc(x):
    return #目标方程（残差平方和的0.5倍）

def r(x):
    return #残差方程。y=y[int(x[n+1])]，t=t[int(x[n+1])]

def Jaccobi(x):
    tar=Function(r)
    mat=np.empty((y.shape[0],n))
    for j in range(0,y.shape[0]):
        for i in range(0,n):
            mat[j][i]=tar.part(i,np.append(x,j))
    return mat

def q(J,R,d):
    return 0.5*muldot(turn(d),turn(J),J,d)+dot(turn(d),dot(turn(J),R))+0.5*dot(turn(R),R)

e=0.001
k=0
v=np.eye(n)
tar=Function(myFunc)
x=np.zeros(n)	#初值点
while tar.norm(np.concatenate((x,t)))>e:
    J=Jaccobi(x)
    A=dot(turn(J),J)+v
    R=np.empty(0)
    for i in range(0,t.shape[0]):
        R=np.append(R,r(np.append(x,i)))
    b=-dot(turn(J),R)
    d=linalg.solve(A,b)
    gamma=(tar.value(x)-tar.value(x+d))/(q(J,R,np.zeros(n))-q(J,R,d))
    if gamma<0.25:
        v*=4
    elif gamma>0.75:
        v/=2
    if gamma>0:
        x+=d
    k+=1
    print(k)
