import scipy.io
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    mat = scipy.io.loadmat('data.mat')
    Y = mat['Y']
    X = mat['X']


    X_train, Y_train = X[:150], Y[:150]
    X_test, Y_test = X[150:], Y[150:]

    clf = SVC(gamma='auto', C = 1, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 2, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 5, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 10, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 20, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 50, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))

    clf = SVC(gamma='auto', C = 100, kernel = 'rbf')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test), len(clf.support_))



    clf.fit(X_train, Y_train)
   # Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(gamma='auto', C = 1, kernel = rbf))])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
