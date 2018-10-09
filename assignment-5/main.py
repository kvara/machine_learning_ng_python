from numpy import *
from functions import *
Theta1 , Theta2 , X , y = load_arrays_from_mat()
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

# lamb = 0
# J , Theta1 , Theta2 = nn_cost_function(Theta1,Theta2,X,y,lamb)
# print(f'test cost function without lambda {J}')

# lamb = 1
# J , Theta1 , Theta2 = nn_cost_function(Theta1,Theta2,X,y,lamb)
# print(f'test cost function with lambda {J}')

initial_theta_1 = rand_initialize_weights(hidden_layer_size, input_layer_size)
initial_theta_2 = rand_initialize_weights(num_labels, hidden_layer_size)

alpha = 1
lamb = 1
Theta1 ,Theta2 = gradient_descent_nn( initial_theta_1 ,initial_theta_2 , X , y , lamb , alpha)

pred = one_vs_all_predict(Theta1,Theta2,X)
accuracy_nn = sum(y.transpose() == pred) / y.shape[0] * 100
print(f'accuracy : {accuracy_nn}')
