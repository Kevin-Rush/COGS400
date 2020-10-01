"""
Kevin Rush
10052650
11kr28

This is my iris perceptron for assignment 1 in COGS 400 2020
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron():

    def __init__(self):        
        self.synaptic_weights = 2*np.random.random((4, 1)) - 1
        self.hidden_synaptic_weights = 2*np.random.random((4, 1)) - 1
        self.learning_rate = 0.01

    def organizeData(self, inputFile):

        with open(inputFile) as f:           #read in the training Data
            lines = [line.split(",") for line in f]

        for i in lines:                             #identify the iris type and change it to a digit, 1 for Versicolor, 0 for setosa, 2 for virginica
            if i[4] == "Iris-setosa\n":
                i[4] = 0
            elif i[4] == "Iris-versicolor\n":
                i[4] = 1
            else:
                i[4] = 2
        
        outputs = [i[4] for i in lines]

        training_inputs = np.array(lines).astype(float)          #convert input and output lists into numpy arrays
        training_outputs = np.array(outputs).T.astype(float)

        training_inputs = np.delete(training_inputs, 4, 1)  #remove the output from the input array

        return training_inputs, training_outputs


    def dotProduct(self, array):
        if len(array) != len(self.synaptic_weights):
            print("Error in Dot Product!!!!")
            return -9999999999999
        else:
            dot = 0
            for i in range(len(array)):
                dot += array[i]*self.synaptic_weights[i]
            return dot
    
    def adjust(self, error, input):
        lower = 1
        if error < 0:
            lower = -1

        for i in range(4):
            self.synaptic_weights[i] += lower*self.learning_rate*input[i]


    def train(self, training_inputs, training_outputs, iterations):
        x = []
        y = []
        j = 0
        for k in range(iterations):                                    
            for i in range(len(training_inputs)):
                output = self.think(training_inputs[i])
                error = training_outputs[i] - output
                y.append(error)
                #print("Error: "+str(error))
                self.adjust(error, training_inputs[i])
                #print("Adjusted")
                #print("New Weights: ")
                #print(self.synaptic_weights)
                j += 1
                x.append(j)

    """
        plt.plot(x, y)

        plt.xlabel('Time')
        plt.ylabel('Error')

        plt.title('Error Over Time')

        plt.show()
    """
    def think(self, inputs):
        output = self.dotProduct(inputs)    #using the sigmoid function as the activation function, send the sigmoid the dot product of inputs agasint the weights
        #print("Input: " + str(inputs))
        #print("Output: "+str(output))
        return output


if __name__ == "__main__":

                 #create a list of ONLY the answers, this way the perceptron will predict and then verify agasint this list

    perceptron = Perceptron()                   #create Perceptron
    
    print("Random synaptic weights: ")          #print initial weights
    print(perceptron.synaptic_weights)
    
    training_inputs, training_outputs = perceptron.organizeData("iris_train.txt")

    perceptron.train(training_inputs, training_outputs, 100) #train the perceptron
    
    print("Synaptic weights post training: ")
    print(perceptron.synaptic_weights)

    print("Time to Guess!")

    testing_inputs, testing_outputs = perceptron.organizeData("iris_test.txt")

    correct = 0
    incorrect = 0

    for i in range(len(testing_inputs)):
        output = perceptron.think(testing_inputs[i])

        if output < 0.5:
            guess = 0
        elif 0.5 < output and output < 1.5:
            guess = 1
        else:
            guess = 2
        
        if guess == testing_outputs[i]:
            print("Correct! - "+str(output))
            correct += 1
        else:
            print("Incorrect - "+str(output))
            incorrect += 1
    
    print("Correct: "+str(correct))
    print("Incorrect: "+str(incorrect))
