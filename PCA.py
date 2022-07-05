import numpy as np
import scipy.io
from scipy import linalg
import matplotlib.pyplot as plt
import DataProcess
import sys
from sklearn.decomposition import PCA

def average_image(data):
    average_image = np.ndarray([])
    count = 0

    for pose in range(len(data)):
        average_image = data[pose] + average_image
        count += 1
    return average_image/count


def center(data):
    average_img = average_image(data)

    for pose in range(len(data)):
            
        data[pose] = data[pose] - average_img

def pca(train, num_dimensions):

    new_train = np.empty_like(train)
    
    for i in range(len(train)):
        new_train[i] = train[i]

    cov = np.cov(np.array(train).T)
   
    # Calculate eigenvalues/eigenvectors
    eigvals, eigvecs = linalg.eigh(cov)

    idx = eigvals.argsort()[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx,:]
    return eigvecs

   
def project(M, train_data):
    vectors = np.ndarray([])
    return np.array([M @ v for v in train_data])


