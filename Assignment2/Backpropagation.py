"""
Back propagation - Assignment 2

Kevin Rush
10052650
11kr28
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class BackpropNN():
    def __init__(self): #file_input_hidden, file_hidden_output):
        #parameters
        self.input_size = 784
        self.hidden_size = 25
        self.output_size = 10
        self.learning_rate = 0.75
        self.a = 0.05
        self.bias = 1
        self.matrix = np.zeros((10, 10))
        self.count = 0
        self.MSE = 9999
        self.avrg_MSE = 0

        self.weights_input_hidden_last_weight_change = np.zeros((self.input_size+1,self.hidden_size), dtype=float)
        self.weights_hidden_output_last_weight_change = np.zeros((self.hidden_size, self.output_size), dtype=float)

        #weights
        #self.IH_test = open(file_input_hidden, "r")
        #self.HO_test = open(file_hidden_output)
        self.weights_input_hidden = 2*np.random.random((self.input_size+1, self.hidden_size))-1 #784x7 weight matrix from input to hidden layer
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size))-1 #7x10 weight matrix from hidden to output layer

        #self.weights_input_hidden = np.empty((self.input_size +  1, self.hidden_size))
        #self.weights_hidden_output = np.empty((self.hidden_size, self.output_size))

        #self.weights_input_hidden = np.ones((self.input_size +  1, self.hidden_size))
        #self.weights_hidden_output = np.ones((self.hidden_size, self.output_size))

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))
    
    def sigmoid_derivaitve (self, input):
        return input * (1 - input)

    def find_index_largest(self, array):
        largest = -1
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i
        
        return index

    def _actual_pred(self, array):
        one_array = np.zeros(10)
        largest = -1
        index = -1
        #print(array.shape)
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i

        one_array[index] = 1

        return one_array
    
    def loss_fn(self, y, output):
        mse = (y - output)**2
        actual_mse = 0
        for i in range(len(mse)):
            actual_mse += mse[i]
        
        self.MSE = actual_mse/len(mse)
        self.avrg_MSE = (self.avrg_MSE+self.MSE)/self.count 


    def feedforward (self, X):                                  #forward propogation through the network

        
        self.z = np.dot(X.T, self.weights_input_hidden)         #dot product of input matrix and first set of weights, returns 784x7 matrix
        self.z2 = self.sigmoid(self.z)                          #activation function, returns the z matrix through the sigmoid function
        
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   #dot product of hidden layer (z2) and second set of weights, returns 784x10 matrix
        output = self.sigmoid(self.z3)
        """
        print("Z: "+str(self.z))
        print("Z2: "+str(self.z2))
        print("Z3: "+str(self.z3))
        print("output: "+str(output))
        print("-------------------------------------------------------------------")
        """
        return output

    def backpropogation(self, X, y, output):                                    #backward propogate through the network
        
        self.loss_fn(y, output)
        self.output_error = (y-output)
        output_sig_der = self.sigmoid_derivaitve(output)
        output_change = self.learning_rate*self.output_error*output_sig_der
        
        output_delta = np.outer(output_change, self.z2) 
        delta_error_sum = np.sum(self.output_error*output_sig_der*self.weights_hidden_output)
        
        z2_change = self.learning_rate*delta_error_sum*self.sigmoid_derivaitve(self.z2)             #Multiplying learning rate, with delta_error_sum and the sigmoid derivative of z2
        z2_delta = np.outer(z2_change, X)

        self.weights_hidden_output += output_delta.T + self.a*self.weights_hidden_output_last_weight_change  #ajdusting the second set (hidden --> output) weights
        self.weights_input_hidden += z2_delta.T + self.weights_input_hidden_last_weight_change

        self.weights_hidden_output_last_weight_change = output_delta.T + self.a*self.weights_hidden_output_last_weight_change
        self.weights_input_hidden_last_weight_change = z2_delta.T + self.weights_input_hidden_last_weight_change   #adjusting first set (input --> hidden) weights

        """
        print("MSE: "+str(self.MSE))
        print("output_eror: "+str(self.output_error))
        print("output sig der: " +str(output_sig_der))        
        print("output delta: "+str(output_delta))
        print("-------------------------------------------------------------------")
        print("sum_delta: "+str(delta_error_sum))
        print("sig_der_z2: "+str(self.sigmoid_derivaitve(self.z2)))
        print("z2 delta"+str(z2_delta))
        print("-------------------------------------------------------------------")
        print("weights:")
        print(self.weights_hidden_output)
        print(self.weights_input_hidden)
        print()
        print(self.weights_hidden_output_last_weight_change)
        print(self.weights_input_hidden_last_weight_change)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        self.count += 1
        if (self.count == 3):
            exit()
        """

    def train(self, X, y):
        array = np.zeros((len(y), 10))
        for i in range(len(y)):
            array[i][y[i]] = 1

        j = 0
        while (j < 5) and (self.MSE > 0.05):
        #for j in range(5):                                  #iteratior
            for i in range(len(X)):
                image = X[i].reshape(784,1)

                #print(image)
                image = np.append(image, [[1]])
                self.count += 1
                output = self.feedforward(image)             #sending one 28x28 matrix that represents an image
                self.backpropogation(image, array[i], output)    #y is a single digit   

            j += 1
            print("J: "+str(j))
            print("MSE: "+str(self.MSE))  
    
    def test(self, X, y):

        correct = 0
        incorrect = 0
        for i in range(len(X)):
            array = np.zeros((len(y), 10))              #one-hot encoding the actual output
            for j in range(len(y)):
                array[j][y[j]] = 1
            image = X[i].reshape(784,1)
            image = np.append(image, [[1]])             #add in the bias
            output = self.feedforward(image)
            output = self._actual_pred(output)           #get an exact number as output
            index_output = self.find_index_largest(output)
            if index_output == self.find_index_largest(array[i]):   #compare if the index's match up
                correct += 1
                #self.matrix[index_output][index_output] += 1
            else:
                incorrect += 1

        return correct, incorrect

if __name__ == "__main__":      #main function to run program and generate output for display purposes
    
    f_hidden_output = "hidden_output_weights.txt"
    f_input_hidden = "input_hidden_weights.txt"
    
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
    train_X = tf.keras.utils.normalize(train_X, axis=1)
    test_X = tf.keras.utils.normalize(test_X, axis=1)
    
    tempIH = NN.weights_input_hidden.copy()
    tempHO = NN.weights_hidden_output.copy()

    NN.train(train_X, train_y)

    """
    for i in range(len(NN.weights_input_hidden)):
        print(NN.weights_input_hidden[i])
    print()
    print("---------------------------------------------------------------------------")
    print()
    for i in range(len(NN.weights_hidden_output)):
        print(NN.weights_hidden_output[i])
    """
    print(NN.test(test_X, test_y))
    #print(NN.matrix)
