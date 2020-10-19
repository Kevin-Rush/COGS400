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
    def __init__(self):
        #parameters
        self.input_size = 784
        self.hidden_size = 20
        self.output_size = 10
        self.learning_rate = 0.01
        self.mse = -1

        #weights
        self.weights_input_hidden = 2*np.random.random((self.input_size, self.hidden_size)) - 1 #784x7 weight matrix from input to hidden layer
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size)) - 1 #7x10 weight matrix from hidden to output layer
    
    def system_check(self):
        print("z: "+str(self.z.shape))
        print("z2: "+str(self.z2.shape))
        print("z3: "+str(self.z3.shape))
        
        print("weights_input_hidden: "+str(self.weights_input_hidden.shape))
        print("weights_hidden_output: "+str(self.weights_hidden_output.shape))

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))
    
    def sigmoid_derivaitve (self, input):
        return input * (1 - input)

    def prediction(self, array):
        largest = -1
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i
        
        return index

    def feedforward (self, X):                                  #forward propogation through the network
        """
        What I need is to find the sum of c*x[i][j]*w[i][j]
        then take the sigmoid of each value as the output of the hidden nodes.
        Then the output of each node goes to all of it's connections from the hidden to the output layer.
        the output layer then takes sum of c*x[i][j]*w[i][j] and outputs the sigmoid as well
        then I just take the largest value's index and that's the guess
        """
        self.z = np.dot(X.T, self.weights_input_hidden)           #dot product of input matrix and first set of weights, returns 784x7 matrix
        self.z2 = self.sigmoid(self.z)                          #activation function, returns the z matrix through the sigmoid function
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   #dot product of hidden layer (z2) and second set of weights, returns 784x10 matrix
        output = self.sigmoid(self.z3)
        #self.system_check()    
        return self.prediction(output.T)

    def backpropogation(self, X, y, output):                                    #backward propogate through the network
        
        #delta_w = c(yj - outputj)*outputj*(1-outputj)*xh

        output_error = (y - output)                                          #error in output
        output_delta = self.learning_rate*output_error*self.sigmoid_derivaitve(output)*self.z3 #how do I get xh???
                
        z2_error = output_delta.dot(self.weights_hidden_output.T)      #z2_error: how much hidden layer weights contribute to output error
        z2_delta = self.learning_rate*z2_error*self.sigmoid_derivaitve(self.z2)*self.z          #applying the derivative of sigmoid to z2 error
        
        self.weights_input_hidden += X.dot(z2_delta)                     #adjusting first set (input --> hidden) weights
        self.weights_hidden_output += self.z2.T.dot(output_delta)          #ajdusting the second set (hidden --> output) weights
        

    def train(self, X, y):
        
        for j in range(1):                                  #iteratior
            for i in range(len(X)):
                image = X[i].reshape(784,1)
                #print(image)
                output = self.feedforward(image)             #sending one 28x28 matrix that represents an image
                
                self.backpropogation(image, y[i], output)    #y is a single digit               


if __name__ == "__main__":      #main function to run program and generate output for display purposes
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images

    """
    print("original input hidden weights")
    for i in range(len(NN.weights_input_hidden)):
        print(NN.weights_input_hidden[i])
    print("original hidden output weights")
    print(NN.weights_hidden_output)
    """
    NN.train(train_X, train_y)
    """
    print("new input hidden weights")
    for i in range(len(NN.weights_input_hidden)):
        print(NN.weights_input_hidden[i])
    print("new hidden output weights")
    print(NN.weights_hidden_output)
"""
    for i in range(10):
        print("Prediction: "+str(NN.feedforward(test_X[i].reshape(784,1))))
        print("Actual: "+str(test_y[i]))

    """
                                                #X is a 6000x28x28
    print("X[i]: "+str(train_X[0]))             #28x28
    print("X[i][i]: "+str(train_X[0][0]))       #28x1
    print("X[i][i][i]: "+str(train_X[0][0][0])) #1
    print("y: "+str(train_y))                   #6000x1

    print("X.shape: "+str(train_X.shape))
    print("X[i].shape: "+str(train_X[0].shape)) 
    print("X[i][i].shape: "+str(train_X[0][0].shape))
    print("X[i][i][i].shape: "+str(train_X[0][0][0].shape))
    print("y.shape: "+str(train_y.shape))

    # display images
    for i in range(3):
        image = test_X[i]
        # plot the sample
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.show()
    """

