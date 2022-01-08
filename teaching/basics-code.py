import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split


"""
Load the diabetes dataset
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
"""
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X.shape)
print(diabetes_y.shape)
print(diabetes_y)


"""
dimension reduction
"""
# pca = decomposition.PCA(n_components=2)
# pca.fit(diabetes_X)
# X_2d = pca.transform(diabetes_X)
# print(X_2d.shape)


"""
plot the projected data
"""
# plt.plot(X_2d[:,0], X_2d[:,1], '.', label="examples")
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()


"""
convert numerical labels to categorical labels
"""
# y_mean = np.mean(diabetes_y)  # 152.13
# print(y_mean)
# y_binary = copy.deepcopy(diabetes_y)
# y_binary[diabetes_y > y_mean] = 1
# y_binary[diabetes_y <= y_mean] = -1
# print(diabetes_y)
# print(y_binary)


"""
np.where will return a tuple of arrays, since y_binary is a 1D
array so returned tuple will contain only one array of indices and
we can get such content by using the subscript 0
"""
# pos_idx = np.where(y_binary == 1)[0]
# neg_idx = np.where(y_binary == -1)[0]


"""
Let's visually check the dataset again with their binary labels.
"""
# plt.plot(X_2d[pos_idx,0], X_2d[pos_idx,1], '+', label="positive examples")
# plt.plot(X_2d[neg_idx,0], X_2d[neg_idx,1], 'x', label="negative examples")
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()


"""
partition data into training set, validation set, and test set
"""
# X_binary = copy.deepcopy(diabetes_X)
# X_train, X_temp, y_train, y_temp = train_test_split(X_binary, y_binary, test_size=0.4, shuffle=True, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
# print(X_binary.shape)
# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)
