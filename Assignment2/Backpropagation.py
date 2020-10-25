"""
Back propagation Model 1 - Assignment 2

Kevin Rush
10052650
11kr28
"""
import tensorflow as tf
import numpy as np

class BackpropNN():
    def __init__(self):
                                        #NN parameters
        self.input_size = 784
        self.hidden_size = 25
        self.output_size = 10
        self.learning_rate = 0.5
        self.alpha = 0.05
        self.bias = 1
                                        #General parameters
        self.count = 0
        self.MSE = 9999
        self.alphavrg_MSE = 0
        self.iterations = 5
        self.accruacy = 0
                                        #weights for momentum
        self.weights_input_hidden_last_weight_change = np.zeros((self.input_size+1,self.hidden_size), dtype=float)
        self.weights_hidden_output_last_weight_change = np.zeros((self.hidden_size, self.output_size), dtype=float)
                                       
                                        #weights 
        self.weights_input_hidden = 2*np.random.random((self.input_size+1, self.hidden_size))-1 
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size))-1 
                                        
                                        #initialize empty train and test comfusion matrix
        self.test_confusion_matrix = np.zeros((self.output_size+1, self.output_size+1), dtype=int)
        self.train_confusion_matrix = np.zeros((self.output_size+1, self.output_size+1), dtype=int)

    def sigmoid(self, input):                   #sigmoid function
        return 1/(1 + np.exp(-input))
    
    def sigmoid_derivaitve (self, input):       #sigmoid derivative function    
        return input * (1 - input)

    def find_index_largest(self, array):        #find's the largest value within an array and returns the index of that value
        largest = -1                            # takes in an array as a parameter and returns an integer
        index = -1                              
        for i in range(len(array)):             
            if largest < array[i]:
                largest = array[i]
                index = i
        
        return index

    def _actual_pred(self, array):              #creates a one-hot encoded array from an array of multiple values
        one_array = np.zeros(10)                #takes in an array of values and returns the "one-hot encoded" version
        largest = -1                            #used to convert output from all nodes into a single node's output
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i

        one_array[index] = 1

        return one_array
    
    def loss_fn(self, y, output):               #calculates the MSE and average MSE of the NN
        mse = (y - output)**2                   #takes in the actual output (y) and the predicted output (output)
        actual_mse = 0                          #used for internal reporting 
        for i in range(len(mse)):
            actual_mse += mse[i]
        
        self.MSE = actual_mse/len(mse)
        self.alphavrg_MSE = (self.alphavrg_MSE+self.MSE)/self.count 

    def feedforward (self, X):                                  #feed's the data forward through the NN
                                                                #takes in a (785,1) matrix representing an image in the MNIST dataset, 
                                                                #returns the output of the NN
        self.z = np.dot(X.T, self.weights_input_hidden)         #dot input.T with the first set of weights
        self.z2 = self.sigmoid(self.z)                          #sigmoid the output from the previous dot product
        
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   #dot the sigmoid'd output from the hidden layer with the 2nd set of weights
        output = self.sigmoid(self.z3)                          #sigmoid the output from the previous dot product

        return output                                           #reutrns the sigmoided output 

    def backpropogation(self, X, y, output):                    #backward propogate through the network
        
        self.loss_fn(y, output)                                 #update loss function, for internal reporting only
        self.output_error = (y-output)                          #E = (dj - yj)
        output_sig_der = self.sigmoid_derivaitve(output)        #f'(yj)
        output_change = self.output_error*output_sig_der        #E*f'(a)
        output_delta = np.outer(output_change, self.z2)         #E*f'(a)*Xh

        error_matrix = self.weights_hidden_output*output_change #E*f'(a)*wjh
        error_matrix = np.sum(error_matrix, axis=1)             #SUM(E*f'(a)*wjh)

        hidden_sig_der = self.sigmoid_derivaitve(self.z2)       #f'(yh)
        errors_multiplied = error_matrix*hidden_sig_der         #SUM(E*f'(a)*wjh)*f'(a)
        z2_delta = np.outer(errors_multiplied, X)               #SUM(E*f'(a)*wjh)*f'(a)*Xi
                                                                #multiply delta_wjh and delta_hi by the learning rate and add it on
                                                                #to the current weight values + momentum
        self.weights_hidden_output += self.learning_rate*output_delta.T + self.alpha*self.weights_hidden_output_last_weight_change  
        self.weights_input_hidden += self.learning_rate*z2_delta.T + self.alpha*self.weights_input_hidden_last_weight_change
                                                                
                                                                #record the weight change from this pass for momentum to be used next iteration
        self.weights_hidden_output_last_weight_change = output_delta.T + self.alpha*self.weights_hidden_output_last_weight_change
        self.weights_input_hidden_last_weight_change = z2_delta.T + self.alpha*self.weights_input_hidden_last_weight_change  

    def matrix_body_update(self, predicted_output, actual_output, test):    #takes in the predicted_output and actual_output values as integers
                                                                            #takes in a boolean to indicate if this is test or training data
                                                                            #uses pred and actual output to update the "body" of the confusion matrix
        if test == False:
            self.train_confusion_matrix[predicted_output][actual_output] +=1
        else:
            self.test_confusion_matrix[predicted_output][actual_output] +=1

    def matrix_edge_update(self, test):                             #updates the "edges" of the confusion matrix based on the data in the body
                                                                    #takes in 1 parameter, a boolean to indicate which matrix to update
        rows = len(self.train_confusion_matrix)-1                   #set numbers of rows to travers, 10
        columns = len(self.train_confusion_matrix[0])-1             #set number of columns to travers, 10

        if test == False:                                           #update training confusion matrix
            index = 0                                               #used to keep track of the "correct prediction" index
            for i in range(rows):
                sum = 0 
                for j in range(columns):
                    sum += self.train_confusion_matrix[i][j]
                percentage = (self.train_confusion_matrix[i][index]/sum)*100    #array is set to dtype=int for visualization purposes 
                                                                                #so I set the percentage to an int as well
                self.train_confusion_matrix[i][rows] = percentage               #update percentage in the last position in the row
                index += 1                                                      #move to next index as "correct prediction"

            index = 0    
            for j in range(columns):                                            #run through each column of the confusion matrix
                                                                                #to update the bottom row of the confusion matrix
                                                                                #essentailly the exact same as the last loop but instead
                                                                                #of going column by row, now going row by column
                sum = 0
                for i in range(rows):
                    sum += self.train_confusion_matrix[i][j]
                    
                percentage = (self.train_confusion_matrix[index][j]/sum)*100
                self.train_confusion_matrix[rows][j] = percentage
                index += 1
            
            total_sum = 0
            correct_sum = 0
            for i in range(rows):                                               #Update the total accuracy in the finale corner of the matrix
                for j in range(columns):
                    total_sum += self.train_confusion_matrix[i][j]              #get the sum of all inputs in the matrix
                    if i == j:
                        correct_sum += self.train_confusion_matrix[i][j]        #sum up all of the "correct predictions" in the matrix
            self.train_confusion_matrix[rows][columns] = correct_sum/total_sum*100
        else:                                                                   #exactly the same as above but for the test matrix instead of the training matrix
            index = 0                                               #used to keep track of the "correct prediction" index
            for i in range(rows):
                sum = 0 
                for j in range(columns):
                    sum += self.test_confusion_matrix[i][j]
                percentage = (self.test_confusion_matrix[i][index]/sum)*100    #array is set to dtype=int for visualization purposes 
                                                                                #so I set the percentage to an int as well
                self.test_confusion_matrix[i][rows] = percentage               #update percentage in the last position in the row
                index += 1                                                      #move to next index as "correct prediction"

            index = 0    
            for j in range(columns):                                            #run through each column of the confusion matrix
                                                                                #to update the bottom row of the confusion matrix
                                                                                #essentailly the exact same as the last loop but instead
                                                                                #of going column by row, now going row by column
                sum = 0
                for i in range(rows):
                    sum += self.test_confusion_matrix[i][j]
                    
                percentage = (self.test_confusion_matrix[index][j]/sum)*100
                self.test_confusion_matrix[rows][j] = percentage
                index += 1
            
            total_sum = 0
            correct_sum = 0
            for i in range(rows):                                               #Update the total accuracy in the finale corner of the matrix
                for j in range(columns):
                    total_sum += self.test_confusion_matrix[i][j]              #get the sum of all inputs in the matrix
                    if i == j:
                        correct_sum += self.test_confusion_matrix[i][j]        #sum up all of the "correct predictions" in the matrix
            self.test_confusion_matrix[rows][columns] = correct_sum/total_sum*100
            

    def train(self, X, y):                                              #Train the NN on the training input X and the training expected output y
        array = np.zeros((len(y), self.output_size))
        for i in range(len(y)):
            array[i][y[i]] = 1
        j = 0
        while j < self.iterations and self.test_confusion_matrix[-1][-1] < 96:  #if accuracy is over 95 percent or the network iterates too much
            for i in range(len(X)):                                             #stop the training
                image = X[i].reshape(self.input_size,1)                         #flatten the current input image
                image = np.append(image, [[self.bias]])                         #add in the bias
                self.count += 1                                                 #keep track of current iteration
                output = self.feedforward(image)                                #feed current input to the network
                self.backpropogation(image, array[i], output)                   #update the network through backpropogation with the predicted output   

                predicted_output = self.find_index_largest(output)              #get predicted output
                actual_output = self.find_index_largest(array[i])               #one-hot encode predicted output
                self.matrix_body_update(predicted_output, actual_output, False) #update training confusion matrix with result
            
            j += 1

    
    def test(self, X, y):                                           #same as training above but with test data

        correct = 0
        incorrect = 0
        for i in range(len(X)):                                     #run through entire test input set
            array = np.zeros((len(y), self.output_size))            #one-hot encoding the actual output
            for j in range(len(y)):
                array[j][y[j]] = 1
            image = X[i].reshape(self.input_size,1)
            image = np.append(image, [[self.bias]])                 #add in the bias
            output = self.feedforward(image)
            
            output = self._actual_pred(output)                      #get an exact number as output
            predicted_output = self.find_index_largest(output)
            actual_output = self.find_index_largest(array[i])
            self.matrix_body_update(predicted_output, actual_output, True)

        self.matrix_edge_update(True)
        return correct, incorrect

            

if __name__ == "__main__":              #main function to run program and generate output for display purposes
    
    NN = BackpropNN()                   #initialize nerual network (NN)

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #get test and train input and output data
    
    train_X = train_X/255               #normalize data
    test_X = test_X/255 

    NN.train(train_X, train_y)          #train network

    print("\nTrain Confusion Matrix: ") #print out confusion network from training data
    print(NN.train_confusion_matrix)

    print(NN.test(test_X, test_y))      #send test data
    print("Test Matrix: ")              #print out confusion network from test data
    print(NN.test_confusion_matrix)
