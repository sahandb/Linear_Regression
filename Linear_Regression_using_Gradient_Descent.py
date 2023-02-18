# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

plt.rcParams['figure.figsize'] = (22.0,13.0)


# Preprocessing Input data
data_train = pd.read_csv('test.csv')
data_test = pd.read_csv('test.csv')

X = data_train.iloc[:, 0]
Y = data_train.iloc[:, 1]

xt = data_test.iloc[:, 0]
xt = data_test.iloc[:, 1]

plt.scatter(X, Y)
plt.show()

# Model
m = 0
c = 0
L = 0.0001  # The learning Rate
cost_previous = 0
epochs = 1000  # The number of iterations to perform gradient descent
n = float(len(X)) # Number of elements in X

def cost(X, Y, theta):
    J=np.dot((np.dot(X,theta) - Y).T,(np.dot(X,theta) - Y))/(2*len(Y))
    return J

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    cost = (1/n)*sum([value**2 for value in (Y - Y_pred)])
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
    cost_previous = cost
print (m, c, cost)
def error_for_line_given_points(c,m,test):
	totalError = 0 	#sum of square error formula
	for i in range (0, len(test)):
		X = test[:, 0]
		Y = test[:, 1]
		totalError += (y-(m*X + c)) ** 2
	return totalError/ float(len(test))

error = error_for_line_given(c,m,xt)

print(error)



# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()