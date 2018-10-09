from numpy import *
from functions import *
X , y = load_arrays_from_mat('ex3data1.mat')

# test part

# theta_t = array([-2, -1, 1, 2])
# lambda_t = 3
# arange_t = transpose((arange(15).reshape(3,5) + 1)) /10
# X_t = concatenate( (ones((5,1)),arange_t), axis=1 )
# y_t = array([1,0,1,0,1])

# print(lr_cost_function(theta_t,X_t,y_t,lambda_t))


lamb = 1
y[y == 10] = 0
X = concatenate((ones((X.shape[0],1)),X),axis=1)

Theta1 , Theta2 = load_arrays_from_mat_nn('ex3weights.mat')
predict_nn = predict(Theta1,Theta2,X)
accuracy_nn = sum(y.transpose() == predict_nn) / y.shape[0] * 100
print(f'accuracy woth neural network : {accuracy_nn}')


all_theta = one_vs_all(X,y,10,lamb)
predict = one_vs_all_predict(all_theta,X)
accuracy_regresion = sum(y.transpose() == predict) / y.shape[0] * 100
print(f'accuracy with regression : {accuracy_regresion}')
