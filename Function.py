import numpy as np

class Function:
    def __init__(self, _f):
        self.fun = _f

    def value(self, val):
        return self.fun(val)

    def part(self, var_index, val):
        a = self.fun(val)
        b = a + 1
        i = 0
        e = 2 ** 10 - 1
        e1 = 2 ** 10
        while 10 ** (-6) < e < e1 or i > -6:
            e1 = e
            a = b
            val_ = list(val)
            val_[var_index] += 10 ** i
            m = self.fun(val_)
            n = self.fun(val)
            b = (m - n) / 10 ** i
            i -= 2
            e = abs(b - a)
        return a

    def part_2(self, x_index, y_index, val):
        return self.__diff_(x_index).__diff_(y_index).value(val)

    def diff(self, val):
        a = self.fun(val)
        b = a + 1
        i = 0
        e = 2 ** 10 - 1
        e1 = 2 ** 10
        while 10 ** (-6) < e < e1 or i > -6:
            e1 = e
            a = b
            val_ = val + 10 ** i
            m = self.fun(val_)
            n = self.fun(val)
            b = (m - n) / 10 ** i
            i -= 2
            e = abs(b - a)
        return a

    def grad(self, val):
        g = np.array(val).astype('float')
        for i in range(0, g.size):
            g[i] = self.part(i, val)
        return np.array(g)

    def __diff_(self, index):
        def diff_f(vals):
            vals_ = list(vals)
            vals_[index] = vals_[index] + 10 ** (-6)
            m = self.fun(vals_)
            n = self.fun(vals)
            return (m - n) / 10 ** (-6)
        return Function(diff_f)

    def hesse(self, val):
        v = np.mat(val)
        G = np.mat(np.dot(v.T, v)).astype('float')
        for i in range(0, v.size):
            for j in range(0, v.size):
                p = self.part_2(i, j, val)
                G[i, j] = p
        G=np.array(G)
        return G
    
    def norm(self, val):
        s = 0
        for x in self.grad(val):
            s += x ** 2
        return np.sqrt(s)
