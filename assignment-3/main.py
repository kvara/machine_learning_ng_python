from numpy import *
from functions import *
from pylab import *

X , y = data_loader('ex2data1.txt')
X = concatenate((ones((X.shape[0],1)),X),axis=1)


# # plot(X[y==1,1],X[y==1,2],'r-x',X[y==0,],X[y==0,2],'g-o')
# # show()


# # პირველი რეგულიზების გარეშე


# # ტესტი

# test_theta = array([-24, 0.2, 0.2])

# cost, grad = cost_function(test_theta, X, y)

# print(f'cost is {cost}')
# print(f'gradient vector is {grad}')

# input('press any key to continue')



# # gradient descent

lamb = 0
num_iters = 4000
alpha = 0.0001

theta = zeros((X.shape[1]))

theta , J_history = gradient_descent(theta,X,y,alpha,num_iters,lamb)
print('-------------------ANSWER-------------------')
prediction = int(sigmoid(array([1,45,85]).dot(theta))*100)
all_prediction = (predict(X,theta) == y).sum()
print(f'predict student with 45 85 points - {prediction}')
print(f'predict students accuracy - {all_prediction}')

input('press any key to continue')

# # რეგულიზებული


# # gradient descent

X , y = data_loader('ex2data2.txt')

lamb = 0
num_iters = 1000
alpha = 0.0001

X = concatenate((ones((X.shape[0],1)),X),axis=1)
theta = zeros((X.shape[1]))
theta , J_history = gradient_descent(theta,X,y,alpha,num_iters,lamb)
print('-------------------ANSWER-------------------')
prediction = int(sigmoid(array([1,45,85]).dot(theta))*100)
all_prediction = (predict(X,theta) == y).sum()
print(f'predict student with 45 85 points - {prediction}')
print(f'predict students accuracy - {all_prediction}')
