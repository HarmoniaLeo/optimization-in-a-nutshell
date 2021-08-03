import numpy as np

def turn(a):	#转置
    if len(a.shape)==1:
        a=a.reshape((1,a.shape[0]))
        return a
    else:
        return np.array(np.mat(a).T)

def dot(a,b):	#点乘
    if len(a.shape)==1:
        a=a.reshape((a.shape[0],1))
    if len(b.shape)==1:
        b=b.reshape((b.shape[0],1))
    res=np.array(np.mat(a)*np.mat(b))
    if res.shape==(1,1):
        return res[0][0]
    else:
        if res.shape[1]==1:
            return res.squeeze()
        else:
            return res

def muldot(*args):	#连乘
    res=args[0]
    for i in range(1,len(args)):
        res=dot(res,args[i])
    return res

def rev(a):	#求逆
    return np.array(np.mat(a).I)
