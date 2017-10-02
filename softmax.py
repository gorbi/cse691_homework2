import numpy as np


class Softmax (object):
    """" Softmax classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random softmax weight matrix to use to compute loss.     #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(inputDim, outputDim))
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        Softmax loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the softmax loss and store to loss variable.                    #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################
        for i in range(x.shape[0]):
            score = x[i].dot(self.W)
            score = score - np.amax(score)
            e_score = np.exp(score)
            sum_e_score = np.sum(e_score)
            target = e_score[y[i]]
            loss += -1 * np.log(target/sum_e_score)
            ds = e_score
            for j in range(score.shape[0]):
                if j == y[i]:
                    continue
                ds[j] = ds[j] / sum_e_score
            ds[y[i]] = target/sum_e_score - 1
            dW += 1/self.W.shape[1] * x[i].T.reshape(x.shape[1], 1).dot(ds.reshape(1, score.shape[0])) + (
                      2 * reg * self.W)
        # take mean of the gradient as it calculated for every image & it shouldn't be the case
        # not sure if taking mean is the perfect way to go
        dW /= x.shape[0]
        # not sure if mean has to be taken for the loss as well, check with TA
        loss /= x.shape[0]
        loss += reg * np.sum(self.W * self.W)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Softmax classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            sample_indices = np.random.choice(np.arange(x.shape[0]), batchSize)
            xBatch = x[sample_indices]
            yBatch = y[sample_indices]

            loss, grad = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.W += -lr * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        yPred = np.argmax(x.dot(self.W), axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        acc = np.mean(self.predict(x) == y) * 100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



