import numpy as np

class srelu:
    def __init__(self, tr, ar, tl, al):
        self.tr = tr
        self.ar = ar
        self.tl = tl
        self.al = al

    def func(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise S shaped Relu of Z, can be modified to relu
        '''
        temp = np.where(Z<=self.tl, self.al*Z, Z)
        temp = np.where(temp>=self.tr, self.ar*temp, temp)
        return temp


    def grad(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise srelu'(z) of Z
        '''
        temp = np.where(Z<=self.tl, self.al, Z)
        temp = np.where(temp>=self.tr, self.ar, 1)
        return temp

    def params_init(self, n, m):
        '''
        Arguements:
        n: int, number of nodes in current layer
        m: int, number of nodes is previous layer

        Returns:
        np array of shape (n,m) with He initialised weights
        '''
        return (np.random.randn(n, m) * np.sqrt(2 / m))

class sigmoid:
    def __init__(self):
        pass

    def func(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise sigmoid of Z
        '''
        return 1 / (1 + np.exp(-Z))

    def grad(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise sigmoid'(z) of Z
        '''
        return ( self.func(Z) * (1 - self.func(Z)) )

    def params_init(self, n, m):
        '''
        Arguements:
        n: int, number of nodes in current layer
        m: int, number of nodes is previous layer

        Returns:
        np array of shape (n,m) with Normalised Xavier initialised weights
        '''
        return (np.random.rand(n, m) * np.sqrt(2 / n + m))

class relu:
    def __init__(self, leaky=0):
        self.leaky = leaky

    def func(self, Z):
        '''
        Arguements:
        Z : numpy array
        leaky : the slope for -Z values, defaults to 0 for relu

        Returns:
        element-wise relu or leaky relu of Z
        '''
        return (np.maximum(self.leaky*Z, Z))

    def grad(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise relu'(z) of Z
        '''
        return np.where(Z<=0, self.leaky, 1)

    def params_init(self, n, m):
        '''
        Arguements:
        n: int, number of nodes in current layer
        m: int, number of nodes is previous layer

        Returns:
        np array of shape (n,m) with He initialised weights
        '''
        return (np.random.randn(n, m) * np.sqrt(2 / m))

class tanh:
    def func(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise tanh(z)
        '''
        return np.tanh(Z)

    def grad(self, Z):
        '''
        Arguements:
        Z : numpy array

        Returns:
        element-wise derivative of tanh(z)
        '''
        return (1 - np.square(Z))

    def params_init(self, n, m):
        '''
        Arguements:
        n: int, number of nodes in current layer
        m: int, number of nodes is previous layer

        Returns:
        np array of shape (n,m) with Normalised Xavier initialised weights
        '''
        return (np.random.rand(n, m) * np.sqrt(2 / n + m))
