import numpy as np
@np.vectorize
## add more activation functions easily by creatin the forward and 
#derrivative function and then add a number for that 

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_der(x):
    x_size = [np.size(x,0),np.size(x,1)]
    y = np.zeros((x_size[0],x_size[1]))
    for i in range(np.size(x,0)):
        for j in range(np.size(x,1)):
            if x[i][j]>0:
                y[i][j] = 1
            else:
                y[i][j] = 0
    return y             

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()    

 
activation_function = sigmoid #### replace with sigmoid or relu 
activation_function_der = sigmoid_der #### replace with sigmoid_der and relu_der 

 


