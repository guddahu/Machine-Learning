import scipy.io
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

def PCA(X1, d):
    X = preprocessing.scale(X1)

    covariance_matrix = np.dot(X.T, X) / len(X)
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

    eigenvecs = []
    for i in range(d):
        eigenvecs.append(np.array(eig_vectors[:, i]))

    x = np.dot(np.array(eigenvecs), X1.T)
    y = np.dot(np.array(eigenvecs).T, x).T

    difference_array = np.subtract(X1, y)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()

    plt.imshow(y[10].reshape(16, 16), cmap="gray")
    plt.title('d = {}, error = {}'.format(d, mse))

    plt.draw()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mat = scipy.io.loadmat('USPS.mat')
    Y = mat['L']
    X1 = mat['A']

    PCA(X1, 10)
    PCA(X1, 50)
    PCA(X1, 100)
    PCA(X1, 200)


