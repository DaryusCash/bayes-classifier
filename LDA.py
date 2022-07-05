import numpy as np

def lda(data, size):
    mean = []
    for subject in range(len(data)):
        avg = sum(data[subject][pose] for pose in range(len(data[subject])))/(len(data[subject]))
        mean.append(avg)

    # Compute within-class scatter matrix
    S_W = np.zeros((len(mean[0]), len(mean[0])))

    for subject in range(len(data)):
        S_i = np.zeros((len(mean[0]), len(mean[0])))

        for pose in range(len(data[subject])):
            x = data[subject][pose].T
            
            mi = mean[subject].T
            S_i += (x - mi) @ (x - mi).T

        S_W += S_i
        ident = np.identity(size) * 0.01

    # Compute between-class scatter matrix
    complete_mean = np.mean([data[subject][pose] for subject in range(len(data)) for pose in range(len(data[subject]))], axis=0) 

    size = len(data[0][0])
    S_B = np.zeros((len(mean[0]), len(mean[0])))
    for i in range(len(mean)):
        cm = complete_mean.T
        mv = mean[i].T 
        S_B += size * (mv - cm) @ (mv - cm).T

    return np.linalg.eigh(np.linalg.inv(S_W + ident) @ (S_B))