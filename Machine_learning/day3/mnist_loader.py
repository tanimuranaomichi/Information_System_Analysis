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

class MNISTImageLoader(object):
    def __init__(self, seed):
        # Load full MNIST, merge train set and test set
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.X = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=3)
        self.y = np.concatenate((y_train, y_test), axis=0)                 
        
        # Shuffle the data
        np.random.seed(seed=seed)
        N = self.X.shape[0]
        shuffle_index = np.random.permutation(N)
        self.X = self.X[shuffle_index, :, :]    
        self.y = self.y[shuffle_index]
        
    def samples(self, N):
        if N < 1:
            return None
        elif N > self.X.shape[0]:
            return (self.X, self.y)
        else:
            return (self.X[:N,:], self.y[:N])
        
class MNISTContestLoader(object):
    def __init__(self):
        # Load full MNIST, merge train set and test set
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        self.X_train = np.expand_dims(X_train, axis=3)
        self.y_train = y_train                 
        
        self.X_test = np.expand_dims(X_test, axis=3)
        self.y_test = y_test
        
    def get_training_set(self):
        return (self.X_train, self.y_train)
        
    def get_testing_set(self):
        return (self.X_test, self.y_test)
        
import requests
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total, unit: x  # If tqdm doesn't exist, replace it with a function that does nothing
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')
    
class KMNISTContestLoader(object):
    
    def __init__(self):
        
        self.filelist = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
                        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
                        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
                        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']
    
        self.download_list()
        
        self.X_train = np.expand_dims(np.load('kmnist-train-imgs.npz')['arr_0'],3)
        self.X_test = np.expand_dims(np.load('kmnist-test-imgs.npz')['arr_0'],3)
        self.y_train = np.load('kmnist-train-labels.npz')['arr_0']
        self.y_test = np.load('kmnist-test-labels.npz')['arr_0']
    
    # Download a list of files
    def download_list(self):
        for url in self.filelist:
            path = url.split('/')[-1]
            r = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)
                    
        print('All dataset files downloaded!')

    def get_training_set(self):
        return (self.X_train, self.y_train)
        
    def get_testing_set(self):
        return (self.X_test, self.y_test)
        
# Main
if __name__ == "__main__":
    mnist_vector_loader = MNISTVectorLoader(42)
    X, y = mnist_vector_loader.samples(10)
    mnist_image_loader = MNISTImageLoader(42)
    X, y = mnist_image_loader.samples(10)