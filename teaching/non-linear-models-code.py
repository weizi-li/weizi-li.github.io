from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


def gen_data(draw=False):
    X, y = make_circles(100, factor=.1, noise=.1, random_state=42)
    pos_idx = np.where(y == 1)
    neg_idx = np.where(y == 0)
    if draw:
        plt.scatter(X[pos_idx,0], X[pos_idx,1], color='red', label="positive examples")
        plt.scatter(X[neg_idx,0], X[neg_idx,1], color='blue', label="negative examples")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()
    return X,y


if __name__ == "__main__":
    X,y = gen_data(draw=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model_linear = SVC(kernel='linear')
    model_linear.fit(X_train, y_train)
    print(model_linear.score(X_test, y_test))

    model_poly = SVC(kernel='poly', degree=10)
    model_poly.fit(X_train, y_train)
    print(model_poly.score(X_test, y_test))

    model_rbf = SVC(kernel='rbf')
    model_rbf.fit(X_train, y_train)
    print(model_rbf.score(X_test, y_test))








