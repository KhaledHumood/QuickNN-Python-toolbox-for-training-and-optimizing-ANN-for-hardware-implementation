import numpy as np
import pickle
import scipy.io
import copy
import NN_activation_function 
from sklearn.preprocessing import MinMaxScaler
 

activation_function = NN_activation_function.activation_function
activation_function2 = NN_activation_function.activation_function_der
softmax = NN_activation_function.softmax 


class NeuralNetwork:
    
    
    def __init__(self, 
                 network_structure,
                 learning_rate,QuantizeLayer,QuantizeLayer_res):
        self.structure = network_structure
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        self.QuantizeLayer = QuantizeLayer
        self.QuantizeLayer_res = (2 ** QuantizeLayer_res) - 1 
        
    def create_weight_matrices(self):
        self.weights_matrices = []
        mu = 0
        sigma = 0.01
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            wm = np.random.normal(mu, sigma,( nodes_out, nodes_in))
            self.weights_matrices.append(wm)
            layer_index += 1
            
    def import_pre_weights(self, weights_pre):
        no_of_layers = len(self.structure)
        no_of_weights = len(weights_pre)
        if no_of_weights == (no_of_layers-1):
            print("predified weights is selected as starting weights")
            layer_index = 1
            while layer_index < no_of_layers:
                self.weights_matrices[layer_index-1] = weights_pre[layer_index-1].T
                layer_index += 1
        else:
            print("Error in predified weights, the code will use random starting weights")
    
    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        no_of_layers = len(self.structure)        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        layer_index = 1
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]
        res_vectors_der = [input_vector] 
        out_vector = []
        out_vector_der = []
        while layer_index < no_of_layers:
            in_vector = res_vectors[layer_index-1]
            x = np.dot(self.weights_matrices[layer_index-1], in_vector)
            if layer_index < no_of_layers - 1:
                out_vector = activation_function(x)
                if self.QuantizeLayer: 
                    scaler = MinMaxScaler((0,self.QuantizeLayer_res))
                    out_vector = scaler.fit_transform(out_vector)
                    out_vector = np.rint(out_vector)
                    out_vector = out_vector/self.QuantizeLayer_res
            else:
                out_vector = softmax(x)
            out_vector_der = activation_function2(x)
            res_vectors.append(out_vector.copy())   
            res_vectors_der.append(out_vector_der.copy())  
            layer_index += 1
        
        layer_index = no_of_layers - 1
        tmp = []
        # The input vectors to the various layers
        output_errors = target_vector - out_vector
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            out_vector_der = res_vectors_der[layer_index]
            in_vector = res_vectors[layer_index-1]
            
            tmp = output_errors * out_vector_der   
            tmp = np.dot(tmp, in_vector.T)
        
            output_errors = np.dot(self.weights_matrices[layer_index-1].T, 
                                    output_errors)  
            
            self.weights_matrices[layer_index-1] += self.learning_rate * tmp
            layer_index -= 1
               
    def train(self, data_array, 
              labels_one_hot_array, test_imgs, test_labels,
              epochs=60, batch = 1000,intermediate_results=False,
              Quantization = False):
        intermediate_weights= []
        pointer = 0
        for epoch in range(epochs):  
            print("Training Epoch = ",epoch)
            for i in range(batch):
                self.train_single(data_array[i+pointer],
                                  labels_one_hot_array[i+pointer]) 
            pointer = pointer + batch
            if (pointer + batch) > len(data_array):
                pointer = 0
            if Quantization:
                if epoch < epochs - 1:
                    print("Quantization start")
                    self.Quantize()
                else:
                    print("Quantization last epoch start")
                    self.Quantize_last(test_imgs,test_labels)
            if intermediate_results:
                intermediate_weights.append(copy.deepcopy(self.weights_matrices))
        print("NN training is done")
        print("")
        return intermediate_weights   
            
    def run(self, input_vector):
        no_of_layers = len(self.structure)
        in_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index-1], 
                        in_vector)
            if layer_index < no_of_layers - 1:
                out_vector = activation_function(x)
                if self.QuantizeLayer: 
                    scaler = MinMaxScaler((0,self.QuantizeLayer_res))
                    out_vector = scaler.fit_transform(out_vector)
                    out_vector = np.rint(out_vector) 
                    out_vector = out_vector/self.QuantizeLayer_res
            else:
                 out_vector = x
            
            # input vector for next layer
            in_vector = out_vector
            layer_index += 1
        return out_vector
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
    
    def Quantize(self):
        
        no_of_layers = len(self.structure)
        layer_index = 1
        W = []
        W_Quan = []
        while layer_index < no_of_layers:
            W = np.transpose(self.weights_matrices[layer_index-1].copy())
            W_size = [np.size(W,0),np.size(W,1)]
            W_abs = np.absolute(W)
            th = np.median(W_abs)
            W_bin = np.zeros((W_size[0],W_size[1]))
            for i in range(W_size[0]):
                for j in range(W_size[1]):
                    if W_abs[i,j]>=th:
                        if W[i,j] > 0:
                            W_bin[i,j] = 1
                        else:
                            W_bin[i,j] = -1
                    else:
                        W_bin[i,j] = 0     
            self.weights_matrices[layer_index-1] = np.transpose(W_bin.copy())
            layer_index += 1
          
    def Quantize_last(self, test_imgs,test_labels):
        no_of_layers = len(self.structure)
        layer_index = 1
        unquantized_weights = copy.deepcopy(self.weights_matrices)
        W = []
        quantized_weights = []
        B_th = []
        # find the base line
        while layer_index < no_of_layers:
            W = np.transpose(unquantized_weights[layer_index-1].copy())
            W_size = [np.size(W,0),np.size(W,1)]
            W_abs = np.absolute(W)
            W_bin = np.zeros((W_size[0],W_size[1]))
            B_th.append(np.median(W_abs.copy()))
            for i in range(W_size[0]):
                for j in range(W_size[1]):
                    if W_abs[i,j]>=B_th[layer_index-1]:
                        if W[i,j] > 0:
                            W_bin[i,j] = 1
                        else:
                            W_bin[i,j] = -1
                    else:
                        W_bin[i,j] = 0
            quantized_weights.append(np.transpose(W_bin.copy()))  
            layer_index += 1
        
        self.weights_matrices = copy.deepcopy(quantized_weights)
        corrects, wrongs = self.evaluate(test_imgs, test_labels)
        base_line = corrects / ( corrects + wrongs)
        step = 0.01
        # optimize the layer
        layer_index = 1
        while layer_index < no_of_layers:
            W = np.transpose(unquantized_weights[layer_index-1].copy())
            W_abs = np.absolute(W)
            W_size = [np.size(W,0),np.size(W,1)]
            W_bin = np.zeros((W_size[0],W_size[1]))
            ####### Step 1: th = th + step ######
            th = B_th[layer_index-1]
            step_optim = base_line
            th_op = th
            for k in range(100):
                th = th + step
                for i in range(W_size[0]):
                    for j in range(W_size[1]):
                        if W_abs[i,j]>=th:
                            if W[i,j] > 0:
                                W_bin[i,j] = 1
                            else:
                                W_bin[i,j] = -1
                        else:
                            W_bin[i,j] = 0
                self.weights_matrices[layer_index-1] = np.transpose(W_bin)
                corrects, wrongs = self.evaluate(test_imgs, test_labels)
                step_acc = corrects / ( corrects + wrongs)
                if step_acc > step_optim:
                    th_op = th
                    step_optim =  step_acc
            ####### Step 2: th = th - step ######
            th = B_th[layer_index-1]
            for k in range(100):
                th = th - step
                for i in range(W_size[0]):
                    for j in range(W_size[1]):
                        if W_abs[i,j]>=th:
                            if W[i,j] > 0:
                                W_bin[i,j] = 1
                            else:
                                W_bin[i,j] = -1
                        else:
                            W_bin[i,j] = 0
                self.weights_matrices[layer_index-1] = np.transpose(W_bin.copy())
                corrects, wrongs = self.evaluate(test_imgs, test_labels)
                step_acc = corrects / ( corrects + wrongs)
                if step_acc > step_optim:
                    th_op = th
                    step_optim =  step_acc
            ########## evaluate the best threshold ##########
            for i in range(W_size[0]):
                for j in range(W_size[1]):
                    if W_abs[i,j]>=th_op:
                        if W[i,j] > 0:
                            W_bin[i,j] = 1
                        else:
                            W_bin[i,j] = -1
                    else:
                        W_bin[i,j] = 0      
            self.weights_matrices[layer_index-1] = np.transpose(W_bin.copy())
            corrects, wrongs = self.evaluate(test_imgs, test_labels)
            step_acc = corrects / ( corrects + wrongs)
            base_line = step_acc
            layer_index += 1