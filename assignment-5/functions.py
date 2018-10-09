import scipy.io as sio
from numpy import *

def sigmoid(Z):
    return 1.0 / (1.0 + exp(-Z))

def sigmoid_gradient(Z):
    return ( 1.0 / (1.0 + exp(-Z)) ) * ( 1.0 - 1.0 / (1.0 + exp(-Z)) )

def load_arrays_from_mat():
    file = sio.loadmat('ex4weights.mat')
    Theta1 = file['Theta1']
    Theta2 = file['Theta2']
    file = sio.loadmat('ex4data1.mat')
    X = file['X']
    y = file['y']
    return Theta1 , Theta2 , X , y

def rand_initialize_weights(L_in, L_out):
    eps = 0.12
    return random.uniform(-eps,eps,(L_in,L_out + 1))

def nn_cost_function( Theta1 , Theta2 , X , y , lamb ):
    m , o = (X.shape[0],1)
    A_1 = concatenate((ones((X.shape[0],1)),X),axis=1)
    A_2 = sigmoid(A_1.dot(transpose(Theta1)))
    A_2 = concatenate((ones((A_2.shape[0],1)),A_2),axis=1)
    A_3 = sigmoid(A_2.dot(transpose(Theta2)))
    temp_y = zeros(shape(A_3))
    J = 0
    
    # აგენერირებს მატრიცას temp_y (m,num_label) , y(m,1) დან , ამ უკანასკნელში წერია ციფრები (1..10)
    # temp_y იწერება 1 y ის შესაბამის მნიშვნელობაზე

    for i in range(0 , m):
        temp_y[i,y[i] - 1] = 1
    J = sum(sum(log(A_3)*(-temp_y) - log(1 - A_3)*(1 - temp_y))) / m
    J += lamb / ( 2 * m ) * ( sum(Theta1[:,1:]**2) + sum(Theta2[:,1:]**2 ) )

    # Backpropagation

    S_3 = A_3 - temp_y

    S_2 = S_3.dot(Theta2[:,1:]) * sigmoid_gradient(A_1.dot(transpose(Theta1)))

    d_2 = S_2.transpose().dot(A_1)
    d_3 = S_3.transpose().dot(A_2)

    Theta1[:,0] = 0
    Theta2[:,0] = 0

    Theta1_grad = d_2 / m + Theta1 * lamb / m
    Theta2_grad = d_3 / m + Theta2 * lamb / m

    return J , Theta1_grad , Theta2_grad

def gradient_descent_nn(Theta1 , Theta2 , X , y , lamb ,alpha):
    num_iter = 400
    for iter in range(0,num_iter):
        J , Grad1 , Grad2 = nn_cost_function(Theta1,Theta2,X,y,lamb)
        Theta1 -= alpha * Grad1
        Theta2 -= alpha * Grad2
    return Theta1 , Theta2
def one_vs_all_predict(Theta1,Theta2,X):
    m , o = (X.shape[0],1)
    A_1 = concatenate((ones((X.shape[0],1)),X),axis=1)
    A_2 = sigmoid(A_1.dot(transpose(Theta1)))
    A_2 = concatenate((ones((A_2.shape[0],1)),A_2),axis=1)
    A_3 = sigmoid(A_2.dot(transpose(Theta2)))
    return A_3.argmax(1) + 1
