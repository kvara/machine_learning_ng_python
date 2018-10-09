from numpy import *

def data_loader(title):
    data = loadtxt(title , delimiter = ',')
    X = data[:,:-1]
    y = data[:,-1]
    return X , y

def sigmoid(Z):
    return 1.0 / (1.0 + exp(-Z))

def cost_function(theta,X,y):
    m = size(y)
    sigmoid_val = sigmoid(X.dot(theta))
    J = (-log(sigmoid_val).dot(y) - log(1 - sigmoid_val).dot(1 - y))  / m
    grad = (sigmoid_val - y).dot(X) / m
    return J , grad

def cost_function_reg(theta,X,y,lamb):
    m = size(y)
    sigmoid_val = sigmoid(X.dot(theta))
    J = (-log(sigmoid_val).dot(y) - log(1 - sigmoid_val).dot(1 - y)) / m + (lamb * pow(theta[1:],2).sum())/ ( 2 * m )
    temp_teta = theta
    temp_teta[0] = 0
    grad = (sigmoid_val - y).dot(X)/m + (lamb/m) * temp_teta
    return J , grad

def gradient_descent(theta,X,y,alpha,num_iters,lamb):
    J_history = zeros((num_iters,1))

    for iter in range(0,num_iters):
        J , grad = cost_function_reg(theta,X,y,lamb)
        theta -=  alpha * grad
        J_history[iter] = J
    return theta , J_history

def predict(theta,X):
    sigmoid_val = sigmoid(theta.dot(X))
    return sigmoid_val > 0.5