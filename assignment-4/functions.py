import scipy.io as sio
from numpy import *

def load_arrays_from_mat(name):
    file = sio.loadmat(name)
    X = file['X']
    y = file['y']
    return X , y

def load_arrays_from_mat_nn(name):
    file = sio.loadmat(name)
    Theta1 = array(file['Theta1'])
    Theta2 = array(file['Theta2'])
    return Theta1 , Theta2

def sigmoid(Z):
    return 1.0 / (1.0 + exp(-Z))

def lr_cost_function(theta,X,y,lamb):
    m = size(y)
    sigmoid_val = sigmoid(X.dot(theta))
    J = (-log(sigmoid_val.transpose()).dot(y) - log((1 - sigmoid_val).transpose()).dot(1 - y)) / m + (lamb * pow(theta[1:],2).sum()) / ( 2 * m )
    temp_teta = theta
    temp_teta[0] = 0
    grad = (X.transpose()).dot(sigmoid_val - y)/m+ (lamb/m) * temp_teta
    return J , grad

def gradient_descent(theta,X,y,alpha,lamb):
    num_iters = 50
    J_history = zeros((num_iters))
    for iter in range(0,num_iters):
        J , grad = lr_cost_function(theta,X,y,lamb)
        theta = theta - alpha * grad
        J_history[iter] = J
    return theta , J_history

def one_vs_all(X,y,num_labels, lamb):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = zeros((n,num_labels))
    alpha = 3
    for i in range(0,num_labels):
        initial_theta = zeros((n,1))
        theta , J_history = gradient_descent(initial_theta,X,y == i,alpha,lamb)
        all_theta[:,i] = theta[:,0]
    return all_theta

def one_vs_all_predict(all_theta,X):
    temp = sigmoid(X.dot(all_theta))
    return temp.argmax(1)

def predict(Theta1,Theta2,X):
    A = sigmoid(X.dot(transpose(Theta1)))
    A = concatenate((ones((A.shape[0],1)),A),axis=1)
    A_2 = sigmoid(A.dot(transpose(Theta2)))
    return A_2.argmax(1) + 1
