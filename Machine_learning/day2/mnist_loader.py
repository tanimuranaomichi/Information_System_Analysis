#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class MNISTVectorLoader(object):
    def __init__(self, seed):
        # Load full MNIST, merge train set and test set
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.X = np.concatenate((X_train, X_test), axis=0)
        self.y = np.concatenate((y_train, y_test), axis=0)
        
        # Transform the features in vector 
        self.X = self.X.astype(np.float32).reshape(-1, 28*28) / 255.0
        self.y = self.y.astype(np.int32)
                
        # Shuffle the data
        np.random.seed(seed=seed)
        N = self.X.shape[0]
        shuffle_index = np.random.permutation(N)
        self.X = self.X[shuffle_index, :]    
        self.y = self.y[shuffle_index]
        
    def samples(self, N):
        if N < 1:
            return None
        elif N > self.X.shape[0]:
            return (self.X, self.y)
        else:
            return (self.X[:N,:], self.y[:N])


# Main
if __name__ == "__main__":
    mnist_vector_loader = MNISTVectorLoader(42)
    X, y = mnist_vector_loader.samples(10)