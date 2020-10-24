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
        #NN parameters
        self.input_size = 784
        self.hidden_size = 15
        self.output_size = 10
        self.learning_rate = 0.1
        self.alpha = 0.05
        self.bias = 1
        #General parameters
        self.count = 0
        self.MSE = 9999
        self.alphavrg_MSE = 0
        self.iterations = 5
        #weights for momentum
        self.weights_input_hidden_last_weight_change = np.zeros((self.input_size+1,self.hidden_size), dtype=float)
        self.weights_hidden_output_last_weight_change = np.zeros((self.hidden_size, self.output_size), dtype=float)
        #weights 
        self.weights_input_hidden = 2*np.random.random((self.input_size+1, self.hidden_size))-1 
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size))-1 
        #initialize empty comfusion matrix
        self.test_confusion_matrix = np.zeros((self.output_size+1, self.output_size+1), dtype=int)
        self.train_confusion_matrix = np.zeros((self.output_size+1, self.output_size+1), dtype=int)

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
        self.alphavrg_MSE = (self.alphavrg_MSE+self.MSE)/self.count 

    def feedforward (self, X):                                  
        
        self.z = np.dot(X.T, self.weights_input_hidden)         
        self.z2 = self.sigmoid(self.z)                          
        
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   
        output = self.sigmoid(self.z3)

        return output

    def backpropogation(self, X, y, output):                          #backward propogate through the network
        
        self.loss_fn(y, output)
        self.output_error = (y-output)
        output_sig_der = self.sigmoid_derivaitve(output)
        output_change = self.output_error*output_sig_der
        output_delta = np.outer(output_change, self.z2) 

        error_matrix = self.weights_hidden_output*output_change
        error_matrix = np.sum(error_matrix, axis=1)

        hidden_sig_der = self.sigmoid_derivaitve(self.z2)
        errors_multiplied = error_matrix*hidden_sig_der
        z2_delta = np.outer(errors_multiplied, X)

        self.weights_hidden_output += self.learning_rate*output_delta.T + self.alpha*self.weights_hidden_output_last_weight_change  
        self.weights_input_hidden += self.learning_rate*z2_delta.T + self.alpha*self.weights_input_hidden_last_weight_change

        self.weights_hidden_output_last_weight_change = output_delta.T + self.alpha*self.weights_hidden_output_last_weight_change
        self.weights_input_hidden_last_weight_change = z2_delta.T + self.alpha*self.weights_input_hidden_last_weight_change  

    def matrix_body_update(self, predicted_output, actual_output, test):
        if test == False:
            self.train_confusion_matrix[predicted_output][actual_output] +=1
        else:
            self.test_confusion_matrix[predicted_output][actual_output] +=1

    def matrix_edge_update(self, test):
        
        if test == False:
            index = 0
            for i in range(len(self.train_confusion_matrix)-1):
                sum = 0 
                for j in range(len(self.train_confusion_matrix[i] - 1) - 1):
                    sum += self.train_confusion_matrix[i][j]
                
                percentage = (self.train_confusion_matrix[i][index]/sum)*100
                self.train_confusion_matrix[i][len(self.train_confusion_matrix)-1] = percentage
                index += 1

            index = 0
            for j in range(len(self.train_confusion_matrix[i] - 1) - 1):
                
                sum = 0
                for i in range(len(self.train_confusion_matrix)-1):
                    sum += self.train_confusion_matrix[i][j]
                    
                percentage = (self.train_confusion_matrix[index][j]/sum)*100
                self.train_confusion_matrix[len(self.train_confusion_matrix)-1][j] = percentage
                index += 1
        else:
            index = 0
            for i in range(len(self.test_confusion_matrix)-1):
                sum = 0 
                for j in range(len(self.test_confusion_matrix[i] - 1) - 1):
                    sum += self.test_confusion_matrix[i][j]
                
                percentage = (self.test_confusion_matrix[i][index]/sum)*100
                self.test_confusion_matrix[i][len(self.test_confusion_matrix)-1] = percentage
                index += 1

            index = 0
            for j in range(len(self.test_confusion_matrix[i] - 1) - 1):
                
                sum = 0
                for i in range(len(self.test_confusion_matrix)-1):
                    sum += self.test_confusion_matrix[i][j]
                    
                percentage = (self.test_confusion_matrix[index][j]/sum)*100
                self.test_confusion_matrix[len(self.test_confusion_matrix)-1][j] = percentage
                index += 1
            
            

    def train(self, X, y):
        array = np.zeros((len(y), self.output_size))
        for i in range(len(y)):
            array[i][y[i]] = 1

        j = 0
        for j in range(self.iterations):                                #iterator
            for i in range(len(X)):
                image = X[i].reshape(self.input_size,1)
                image = np.append(image, [[self.bias]])
                self.count += 1
                output = self.feedforward(image)             #sending one 28x28 matrix that represents an image
                self.backpropogation(image, array[i], output)    #y is a single digit   

                #one_hot_output = self._actual_pred(output)           #get an exact number as output
                predicted_output = self.find_index_largest(output)
                actual_output = self.find_index_largest(array[i])
                self.matrix_body_update(predicted_output, actual_output, False)

            j += 1
            print("J: "+str(j))
            print("MSE: "+str(self.MSE))
            self.matrix_edge_update(False)
            print(self.train_confusion_matrix)
        
        self.matrix_edge_update(False)
  
    
    def test(self, X, y):

        correct = 0
        incorrect = 0
        for i in range(len(X)):
            array = np.zeros((len(y), self.output_size))              #one-hot encoding the actual output
            for j in range(len(y)):
                array[j][y[j]] = 1
            image = X[i].reshape(self.input_size,1)
            image = np.append(image, [[self.bias]])             #add in the bias
            output = self.feedforward(image)
            
            output = self._actual_pred(output)           #get an exact number as output
            predicted_output = self.find_index_largest(output)
            actual_output = self.find_index_largest(array[i])
            self.matrix_body_update(predicted_output, actual_output, True)
            #print(self.test_confusion_matrix)
            if predicted_output == actual_output:   #compare if the index's match up
                correct += 1
            else:
                incorrect += 1

        return correct, incorrect

            

if __name__ == "__main__":      #main function to run program and generate output for display purposes
    
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
    train_X = train_X/255 #tf.keras.utils.normalize(train_X, axis=1)
    test_X = test_X/255 #tf.keras.utils.normalize(test_X, axis=1)
    
    tempIH = NN.weights_input_hidden.copy()
    tempHO = NN.weights_hidden_output.copy()

    NN.train(train_X, train_y)

    print("Train Confusion Matrix: ")
    print(NN.train_confusion_matrix)
    print(NN.test(test_X, test_y))
    print("Test Confusion Matrix: ")
    print(NN.test_confusion_matrix)
