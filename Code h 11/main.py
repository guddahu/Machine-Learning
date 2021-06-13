# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
import math
def least_squares(feature, target):
    """
        Compute the model vector obtained after MLE
        w_star = (X^T X)^(-1)X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :return: w_star (d+1)x1 model vector
        """
    #TODO
    w_star = np.dot(np.linalg.inv(np.dot(feature.T, feature)), np.dot(feature.T, target))
    return w_star
def compute_objective_value(feature, target, model):
    # Compute MSE
    mse = mean_squared_error(target, np.dot(feature, model))
    return mse
def mean_squared_error(true_label, predicted_label):
    """
        Compute the mean square error between the true and predicted labels
        :param true_label: Nx1 vector
        :param predicted_label: Nx1 vector
        :return: scalar MSE value
    """
    #TODO

    mse = np.sqrt(np.sum((true_label - predicted_label)**2))/len(true_label)
    return mse


def gradient_descent(feature, target, step_size, max_iter, lam = 1e-17):
    w = least_squares(feature, target)
    w = np.zeros((feature.shape[1],1))
    n = step_size
    iter = []
    objective_value = []


    for i in range(max_iter):
        iter.append(i)
        # Compute gradient
        gradient = np.dot(feature.T,np.dot(feature, w) - target) + lam * w
        # Update the model
        w = w - (n * gradient)
        # Compute the error (objective value)
        objective_value.append(compute_objective_value(feature,target, w))

    return w, objective_value, iter

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    file_ziptrain = open("ZipDigits.train", "r")
    LINES = file_ziptrain.readlines()



    one = []
    five = []
    for line in LINES:
        line_by_space = line.split()
        for i in range(len(line_by_space)):
            line_by_space[i] = float(line_by_space[i])
        if float(line_by_space[0]) == 1:
            line_by_space = np.array(line_by_space[1:])
            one.append(line_by_space.reshape(16,16))
        if float(line_by_space[0]) == 5:
            line_by_space = np.array(line_by_space[1:])
            five.append(line_by_space.reshape(16,16))

    one_intensity = []
    for i in range(len(one)):
        one_intensity.append((sum(sum(one[i]))))

    five_intensity = []
    for i in range(len(five)):
        five_intensity.append((sum(sum(five[i]))))

    one_symetry = []
    for i in range(len(one)):
        one_symetry.append(-1*sum(sum(np.abs(one[i] - np.fliplr(one[i]))))*10)

    five_symetry = []
    for i in range(len(five)):
        five_symetry.append(-1*sum(sum(np.abs(five[i] - np.fliplr(five[i]))*10)))


    feature = np.zeros((len(one) + len(five), 3))
    target = np.zeros((len(one) + len(five), 1))
    for i in range(len(one)):
        feature[i][0] = one_symetry[i]
        feature[i][1] = one_intensity[i]
        feature[i][2] = 1
        target[i][0] = 1
    for i in range(len(five)):
        feature[i+len(one)][0] = five_symetry[i]
        feature[i+len(one)][1] = five_intensity[i]
        feature[i+len(one)][2] = 1
        target[i+len(one)][0] = -1


    lr = LogisticRegression(C=1e5)
    lr.fit(feature, target)

    w, objective_value, iter = gradient_descent(np.array(feature), target, 0.00000000001, 10000)
    w0 = w[2][0]
    w1 = w[1][0]
    w2 = w[0][0]

    # w0 = lr.coef_[0][2]
    # w1 = lr.coef_[0][1]
    # w2 = lr.coef_[0][0]


    x1 = feature[:, 0]
    y = (-(w2*x1 + w0)/w1)

    error = 0
    for i in range(len(feature)):
        error += np.log(1 + np.exp(-target[i] * (w0 + w1*feature[i][0] + w2*feature[i][1] )))
    error = error/len(feature)

    print(error)




    feature1 = np.zeros((len(one) + len(five), 10))
    target1 = np.zeros((len(one) + len(five), 1))
    for i in range(len(one)):
        feature1[i][0] = one_symetry[i]
        feature1[i][1] = one_intensity[i]
        feature1[i][2] = one_symetry[i]*one_symetry[i]
        feature1[i][3] = one_intensity[i]*one_symetry[i]
        feature1[i][4] = one_intensity[i]*one_intensity[i]
        feature1[i][5] = one_symetry[i]*one_symetry[i]*one_symetry[i]
        feature1[i][6] = one_symetry[i]*one_symetry[i]*one_intensity[i]
        feature1[i][7] = one_symetry[i]*one_intensity[i]*one_intensity[i]
        feature1[i][8] = one_intensity[i]*one_intensity[i]*one_intensity[i]
        feature1[i][9] = 1
        target[i][0] = 1
    for i in range(len(five)):
        feature1[i + len(one)][0] = one_symetry[i]
        feature1[i + len(one)][1] = one_intensity[i]
        feature1[i + len(one)][2] = one_symetry[i]*one_symetry[i]
        feature1[i + len(one)][3] = one_intensity[i]*one_symetry[i]
        feature1[i + len(one)][4] = one_intensity[i]*one_intensity[i]
        feature1[i + len(one)][5] = one_symetry[i]*one_symetry[i]*one_symetry[i]
        feature1[i + len(one)][6] = one_symetry[i]*one_symetry[i]*one_intensity[i]
        feature1[i + len(one)][7] = one_symetry[i]*one_intensity[i]*one_intensity[i]
        feature1[i + len(one)][8] = one_intensity[i]*one_intensity[i]*one_intensity[i]
        feature1[i + len(one)][9] = 1
        target[i+len(one)][0] = -1


    w_1, objective_value1, iter1 = gradient_descent(np.array(feature1), target, 0.0000000000000000000001, 5000)

    error1 = 0
    for i in range(len(feature1)):
        error1 += np.log(1 + np.exp(-target[i] * (w_1[9][0] + w_1[8][0]*feature1[i][0] + w_1[7][0]*feature1[i][1] + w_1[6][0]*feature1[i][2]
                                                  + w_1[5][0]*feature1[i][3] + w_1[4][0]*feature1[i][4] + w_1[3][0]*feature1[i][5]
                                                  + w_1[2][0]*feature1[i][6] + w_1[1][0]*feature1[i][7] + w_1[0][0]*feature1[i][8])))
    error1 = error/len(feature1)

    print(error,error1)
    plt.plot(x1,y)
    plt.scatter(one_symetry ,one_intensity, marker = 'o', color = 'blue', facecolors='none', label = 'one')
    plt.scatter(five_symetry,five_intensity, marker = 'x', color = 'red', label = 'five')
    plt.xlabel("Symetry")
    plt.ylabel("intensity")
    plt.legend(loc="upper left")
    plt.title('Training Data')
    plt.show()


    #
    #
    plt.imshow(one[1],cmap="gray",vmin = -1, vmax = 1)
    #plt.imshow(five[1],cmap="gray",vmin = -1, vmax = 1)

#
    plt.draw()
    plt.show()









# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
