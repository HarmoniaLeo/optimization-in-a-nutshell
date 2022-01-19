from Function import Function	#定义法求导工具
import numpy as np
from scipy import linalg
from lagb import *	#线性代数工具库

n=4	#待定系数数
y=np.array([0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
t=np.array([4,2,1,0.5,0.25,0.1670,0.1250,0.1000,0.0833,0.0714,0.0625])
#样本列表


def myFunc(x):  #目标函数（残差平方和的0.5倍）
    return 1/2*np.sum(np.square(y-x[0]*(t**2+x[1]*t)/(t**2+x[2]*t+x[3])))

def r(x):   #残差函数
    yi = y[int(x[n])]
    ti = t[int(x[n])]
    fxt = x[0]*(ti**2+x[1]*ti)/(ti**2+x[2]*ti+x[3]) #原函数
    return yi - fxt

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
print(x)