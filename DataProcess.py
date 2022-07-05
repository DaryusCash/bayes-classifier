import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io 


def task_2_split(test_ratio=None):
    Ns = 200 
    face = scipy.io.loadmat(r"../Data/data.mat")['face']
    face_n = [face[:,:,3*n] for n in range(Ns)] # neutral
    face_x = [face[:,:,3*n+1] for n in range(Ns)] # expression
    face_il = [face[:,:,3*n+2] for n in range(Ns)] # illumination variation

    data = []
    labels = []
    for subject in range(Ns):
        # neutral face: label 0
        data.append(face_n[subject].reshape(-1))
        labels.append(0)
        #expression: label 1
        data.append(face_x[subject].reshape(-1))
        labels.append(1)

    # Split to train and test data
    N = int( (1-test_ratio)*len(data) )
    idx = np.arange(len(data))
    random.shuffle(idx)
    train_data = [data[i] for i in  idx[:N]]
    train_labels = [labels[i] for i in  idx[:N]]
    test_data = [data[i] for i in  idx[N:]]
    test_labels = [labels[i] for i in  idx[N:]]

    
    return (train_data, train_labels, test_data, test_labels)

def task_1_split(test_ratio=None):
    illum = scipy.io.loadmat(r"../Data/illumination.mat")['illum']

    # # Convert the dataset in data vectors and labels for subject identification
    data = []
    labels = []
    for subject in range(illum.shape[2]):
        for image in range(illum.shape[1]):
            data.append(illum[:,image,subject])
            labels.append(subject)

    # Split to train and test data    
    N = int( (1-test_ratio)*len(data) )
    idx = np.arange(len(data))
    random.shuffle(idx)
    train_data = [data[i] for i in  idx[:N]]
    train_labels = [labels[i] for i in  idx[:N]]
    test_data = [data[i] for i in  idx[N:]]
    test_labels = [labels[i] for i in  idx[N:]]

    return (train_data, train_labels, test_data, test_labels)


def list_to_dict(data, labels):
    data_dict = {}

    for i in range(len(data)):

        if labels[i] not in data_dict.keys():
            
            data_dict[labels[i]] = []
        
        data_dict[labels[i]].append(data[i])
    
    return data_dict

