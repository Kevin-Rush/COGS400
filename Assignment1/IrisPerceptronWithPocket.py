"""
Kevin Rush
10052650
11kr28

This is my iris perceptron for assignment 1 Part B (Pocket) in COGS 400 2020
"""

import numpy as np

class Perceptron():                     #Creating the perceptron class

    def __init__(self):                 #initialize perceptron properties, synaptic weights  and the learning rate
        self.synaptic_weights = 2*np.random.random((5, 1)) - 1  #random weight values to start with
        self.learning_rate = 0.01       #I chose a small learning rate because it provided the best results
        self.bias = 1
        self.pocket_weights = self.synaptic_weights.copy()
        self.best_run = -1
        self.current_run = 0
        self.total_incorrect = 0
        self.total_correct = 0

    def organizeData(self, inputFile):  #This is a function to read in a file, put all the data into a useable input and output array

        with open(inputFile) as f:           #read in the Data
            lines = [line.split(",") for line in f]

        for i in lines:                        #identify the iris type and change it to a digit, 0 for setosa, 1 for Versicolor, 2 for virginica
            if i[4] == "Iris-setosa\n":
                i[4] = 0
            elif i[4] == "Iris-versicolor\n":
                i[4] = 1
            else:
                i[4] = 2
        np.random.shuffle(lines)
        outputs = [i[4] for i in lines]         #fillin only the last index of all read in lines to create the output array
        for i in lines:                         #fill in the finale slot of all inputs with the bias
            i[4] = self.bias
        
        training_inputs = np.array(lines).astype(float)          #convert input and output lists into numpy arrays as float types
        training_outputs = np.array(outputs).T.astype(float)
        return training_inputs, training_outputs        #return useable arrays


    def dotProduct(self, array):                    #calcualte the dot product of an array with the synaptic weights 
        if len(array) != len(self.synaptic_weights):    #just a check to make sure the arrays are compatable 
            print("Error in Dot Product!!!!")
            return -9999999999999
        else:
            dot = 0
            for i in range(len(array)):
                dot += array[i]*self.synaptic_weights[i]
            return dot                              
    
    def adjust(self, error, input):             #adjusts the weights of the synaptic weight
        lower = 1                           #if the guess was "lower" than the actual output then the error should be positive
        if error < 0:                       
            lower = -1                      #if the guess was higher than the actual output then the error should be negative
                            #I used the lower feature to manage the appropriate adjustments if y < D and if D < y
        for i in range(4):
            self.synaptic_weights[i] += lower*self.learning_rate*input[i]   #update all the weights


    def train(self, training_inputs, training_outputs, iterations): #train our data!

        for k in range(iterations):                         #iterate through the data multiple times for training
            for i in range(len(training_inputs)):           #run through each line of the training input individually
                output = np.dot(training_inputs[i], self.synaptic_weights)     #get an output from our think function

                if output < 0.5:                    #if the output is less than 0.5, then I commit guess to being setosa
                    guess = 0
                elif 0.5 < output and output < 1.5: #if the output is greater than 0.5 but less than 1.5, then I commit guess to being versicolor
                    guess = 1
                else:
                    guess = 2 
                
                if guess == training_outputs[i]:                 #Check if we were correct and track the predictions
                    self.total_correct += 1
                    self.current_run += 1
                    if self.current_run > self.best_run:
                        self.best_run = self.current_run
                        self.pocket_weights = self.synaptic_weights.copy()
                else:
                    self.total_incorrect += 1
                    self.current_run = 0
                
                error = training_outputs[i] - output        #calculate the error
                self.adjust(error, training_inputs[i])      #adjust weights accordingly
            self.synaptic_weights = self.pocket_weights.copy()
                    
                

                


    def think(self, inputs):
        output = np.dot(inputs, self.synaptic_weights)    #use the dot product to calculate our guess

        if output < 0.5:                    #if the output is less than 0.5, then I commit guess to being setosa
            guess = 0
        elif 0.5 < output and output < 1.5: #if the output is greater than 0.5 but less than 1.5, then I commit guess to being versicolor
            guess = 1
        else:
            guess = 2                        #if the output is greater than 1.5, then I commit guess to being virginica
 
        return guess


if __name__ == "__main__":

    perceptron = Perceptron()                   #create Perceptron object
    
    print("Random synaptic weights: ")          #print initial weights for monitoring purposes
    print(perceptron.synaptic_weights)
    
    training_inputs, training_outputs = perceptron.organizeData("iris_train.txt")
    perceptron.train(training_inputs, training_outputs, 10) #train the perceptron, runs through the data 100 times
    
    print("Synaptic weights post training: ")   #print finale weights for monitroing purposes
    print(perceptron.synaptic_weights)

    print(perceptron.total_correct)
    print(perceptron.total_incorrect)
    print(perceptron.best_run)


    print("Time to Guess!")

    testing_inputs, testing_outputs = perceptron.organizeData("iris_test.txt") #load testing data

    correct = 0             #counters to keep track of correct vs incorrect
    incorrect = 0

    for i in range(len(testing_inputs)):                 #run through each line of the testing input and see if we guess correctly
        output = perceptron.think(testing_inputs[i])     #get output

        if output == testing_outputs[i]:                 #Check if we were correct and track the predictions
            correct += 1
        else:
            incorrect += 1
    
    print("Correct: "+str(correct))                 
    print("Incorrect: "+str(incorrect))



