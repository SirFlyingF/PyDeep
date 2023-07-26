#import activation
import numpy as np
import matplotlib.pyplot as plt

'''
NNet with batch gradient desent
layer1 (dims[0]) is X, layer L+1 is output layer
'''

'''
TO DO:
-implment input norm
-implement optimisers: move entire bckprop to seperate classes
-implement batch norm
-convert into regression problem
'''

class mlp:
    def __init__(self, dims, acts):
        '''
        Arguements:
        dims: list, format-> [nx, n1, n2...nL], nL is the size of last hidden layer
        acts: list of object of classes from activations module or custom compatible class, format-> [g1, g2, g3...gL+1], g1 is g() for first hidden layer, gL+1 is for output layer
        '''

        #copy acts into an instance variable, first element 0 to allign with dims
        self.acts = [0] + acts
        self.params = {}
        dims.append(1)#append 1 neuron for output layer
        for idx in range(len(dims)-1): #-1 for idx correction
            layer=idx+1
            self.params['W' + str(layer)] = self.acts[layer].params_init(dims[layer], dims[layer-1]) #(nl x nl-1)
            self.params['b' + str(layer)] = np.zeros((dims[layer], 1)) #(nl, 1)

        #cache to store A and Z
        self.cache = {}

        #len(dims) - output layer - input layer = hidden layers
        self.L = len(dims) - 2


    def fwdprop(self, train_set):
        '''
        Arguements:
        train_set : np array, shape=(feature_vector.shape, num_examples) = (dims[0], m)
        (Future) Activations : list of callables for layer-wise activations

        Returns:
        A : np array, activation of the last hidden layer
        '''
        A = train_set
        for idx in range(self.L+1):
            layer = idx+1
            #Z = W[l].A[l-1] + b
            #A[l] = g(Z)
            Z = np.dot(self.params['W' + str(layer)], A) + self.params['b' + str(layer)]
            A = self.acts[layer].func(Z)
            self.cache['A' + str(layer)] = A
            self.cache['Z' + str(layer)] = Z

        return A

    def bckprop_output(self, y_hat, labels, learning_rate, m):
        '''
        Arguements:
        y_hat : np array, activation of output layer
        labels : np array, ground truth labels
        learning_rate : float
        m : int, number of traning examples in (mini)batch

        Returns:
        dZ_output : np array, dCost/dZ[output] -> delta1
        -Also updates weights for output layer
        '''
        #dy_hat -> dCost/dy_hat
        dy_hat = (y_hat - labels) / (y_hat * (1 - y_hat))

        #dZ_output -> dCost/dZ_output = (dCost/dy_hat) * grad[L+1]()
        dZ_output = dy_hat * self.acts[self.L+1].grad(self.cache['Z' + str(self.L+1)])

        #weights update for output layers
        dW = (1/m) * np.dot(dZ_output, self.cache['A' + str(self.L)].T)
        db = (1/m) * np.sum(dZ_output, keepdims=True, axis=1)

        assert dW.shape == self.params['W' + str(self.L+1)].shape, "DEVERR: bckprop"
        assert db.shape == self.params['b' + str(self.L+1)].shape, "DEVERR: bckprop"
        self.params['W' + str(self.L+1)] -= learning_rate * dW
        self.params['b' + str(self.L+1)] -= learning_rate * db

        #dZ_output becomes delta1 for further bckprop
        return dZ_output


    def bckprop_hidden(self, delta, learning_rate, m):
        '''
        Arguements:
        delta : np array, calculated gradients from output layer
        learning_rate : float
        m : int, number of traning examples in (mini)batch

        Returns:
        None
        -Calculates and updates weights for all hidden layers
        '''
        for idx in range(self.L -1):
            layer = self.L - idx #because we're going backwards

            #delta2 = W[l+1] dot delta1 * g'(Z[l])
            delta = np.dot(self.params['W' + str(layer+1)].T, delta) * self.acts[layer].grad(self.cache['Z' + str(layer)])

            #weight update for current layer
            dW = (1/m) * np.dot(delta, self.cache['A' + str(layer-1)].T)
            db = (1/m) * np.sum(delta, axis=1, keepdims=True)
            self.params['W' + str(layer)] -= learning_rate * dW
            self.params['b' + str(layer)] -= learning_rate * db

        return None

    def train(self, learning_rate, train_set, labels, epoch, batch_size, plot=False, verbose=True):
        '''
        Arguements:
        learning_rate: float,
        train_set: np array, dimentions (feature_vector_len, num_examples)
        labels: np array, ground truth, dimentions(1, num_examples)
        epoch: positive int, number of timesteps
        batch_size: int, size of mini batch, must be less than train_set[0].shape && >= 1
        plot: bool, plots cost vs iter graph if True, defaults to False
        verbose: bool, prints cost after every 100 iter if True, defaults to True
        '''
        assert train_set[0].shape == labels[0].shape, "train_set - label shape mismatch"
        assert batch_size >= 1 or batch_size <= train_set[0].shape[0], "invalid size of mini batch"
        assert epoch >= 1, "number of iterations must be positive"
        CostPlot = []

        #mini batch, if batch size = 1 : reduces to stochastic
        if batch_size < train_set[0].shape[0]:
            num_batch = (len(train_set[0]) // batch_size)
            partial_batch_size = len(train_set) - num_batch*batch_size
            m = batch_size
        else:
            #batch gradient descent
            partial_batch_size = 0
            m = train_set[0].shape[0]
            num_batch = 1

        for epoch in range(epoch+1):
            for batch in range(num_batch):
                X = train_set[:, m*batch : (1+batch)*m]
                Y = labels[:, m*batch : (1+batch)*m]

                #fwd prop
                y_hat = self.fwdprop(X)
                assert y_hat.shape == Y.shape, "Unexpected shape of y_hat"

                #Cost and Loss -> Binary cross entropy
                Loss = ( Y * np.log(y_hat) + (1 - Y) * np.log(1-y_hat) )
                Cost = np.sum(Loss) * (-1 / m)
                CostPlot.append(Cost)

                #bck prop
                delta = self.bckprop_output(y_hat, Y, learning_rate, m)
                self.bckprop_hidden(delta, learning_rate, m)

            if partial_batch_size > 0:
                X = train_set[:, num_batch*m :]
                Y = labels[:, num_batch*m :]

                #fwd prop
                y_hat = self.fwdprop(X)
                assert y_hat.shape == Y.shape, "Unexpected shape of y_hat"

                #Cost and Loss -> Binary cross entropy
                Loss = (Y*np.log(y_hat) + (1-Y)*np.log(1-y_hat))
                Cost = np.sum(Loss) * (-1 / partial_batch_size)
                CostPlot.append(Cost)

                #bck prop
                delta = self.bckprop_output(y_hat, Y, learning_rate, partial_batch_size)
                self.bckprop_hidden(delta, learning_rate)

            if (epoch % 10 == 0) and verbose:
                print(f"Cost after epoch {epoch} = {Cost}")

        if plot:
            plot = plt.plot(CostPlot)
            plt.show()


    def test(self, test_set, labels, pred=False):
        '''
        Arguements:
        test_set: np array, dimentions(feature_vector_len, num_examples)
        labels: np array, dimentions(1, num_examples), not required if pred=True
        pred: Bool, prediction mode or testing mode, deafults to False

        Returns:
        if pred is False: returns accuracy against labels
        if pred is True: returns activations of output layer without floor func
        '''
        #fwdprop
        y_hat = self.fwdprop(test_set)
        if pred:
            return y_hat

        #calculate accuracy
        assert labels is not None, "labels required to use test mode"
        y_hat = [y_hat[idx] // 0.5 for idx in range(len(y_hat))]
        correct = np.where(y_hat == labels, 1, 0)
        perc = (np.sum(correct) / len(labels[0])) * 100
        return perc
