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
        self.input_size = 28
        self.hidden_size = 7
        self.output_size = 10
        self.all_errors = []

        #weights
        self.weights_input_hidden = 2*np.random.random((self.input_size, self.hidden_size)) - 1 #28x7 weight matrix from input to hidden layer
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size)) - 1 #7x9 weight matrix from hidden to output layer
    
    def feedforward (self, X):                                  #forward propogation through the network
        
        self.z = np.dot(X, self.weights_input_hidden)           #dot product of input matrix and first set of weights
        self.z2 = self.sigmoid(self.z)                          #activation function
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   #dot product of hidden layer (z2) and second set of weights 
        output = self.sigmoid(self.z3)
        #self.system_check()
        return output

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

    def backpropogation(self, X, y, output):                            #backward propogate through the network
        self.output_error = y - output                                  #error in output
        self.output_delta = self.output_error*self.sigmoid_derivaitve(output)

        self.z2_error= self.output_delta.dot(self.weights_hidden_output.T)               #z2_error: how much hidden layer weights contribute to output error
        self.z2_delta = self.z2_error*self.sigmoid_derivaitve(self.z2)  #applying the derivative of sigmoid to z2 error

        self.weights_input_hidden += X.T.dot(self.z2_delta)                             #adjusting first set (input --> hidden) weights
        self.weights_hidden_output += self.z2.T.dot(self.output_delta)                   #ajdusting the second set (hidden --> output) weights

    def train(self, X, y):
        for i in range(len(X)):
            output = self.feedforward(i)
            self.backpropogation(X[i], y[i], output)


if __name__ == "__main__":      #main function to run program and generate output for display purposes
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
    print("Initial Weights for input to hidden: "+str(NN.weights_input_hidden))
    print("Initial Weights for hidden to output: "+str(NN.weights_hidden_output))

    NN.train(train_X, train_y)

    print("New Weights for input to hidden: "+str(NN.weights_input_hidden))
    print("New Weights for hidden to output: "+str(NN.weights_hidden_output))


    # display images
    for i in range(0):
        image = train_X[i]
        # plot the sample
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.show()



