from sklearn.datasets import make_regression
from matplotlib import pyplot as plt


""" 
perform gradient descent for one step
"""
def gradient_one_step(X, y, slope, intercept, learning_rate):
    slope_gradient = 0
    intercept_gradient = 0
    N = len(y)
    for i in range(N):
        slope_gradient += -(1 / N) * X[i] * (y[i] - (slope * X[i] + intercept))
        intercept_gradient += -(1 / N) * (y[i] - (slope * X[i] + intercept))
    new_slope = slope - (learning_rate * slope_gradient)
    new_intercept = intercept - (learning_rate * intercept_gradient)
    return [new_slope, new_intercept]


""" 
the gradient descent algorithm
"""
def gradient_descent(X, y, init_slope, init_intercept, learning_rate, num_iter):
	slope = init_slope
	intercept = init_intercept
	for i in range(num_iter):
		slope, intercept = gradient_one_step(X, y, slope, intercept, learning_rate)
	return [slope, intercept]


""" 
generate a dataset
"""
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
plt.scatter(X,y,label="training data")


""" 
set initial parameters
"""
learning_rate = 0.01
init_slope = 0
init_intercept = 0
num_iter = 1000


""" 
run the gradient descent algorithm and check the results
"""
[slope, intercept] = gradient_descent(X, y, init_slope, init_intercept, learning_rate, num_iter)
plt.plot(X, slope*X + intercept, color="red", label="fitted line")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

