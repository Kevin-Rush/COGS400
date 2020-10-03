"""
Kevin Rush
10052650
11kr28

This is my iris perceptron for assignment 1 Part A (No Pocket) in COGS 400 2020
"""

import numpy as np

class Perceptron():                     #Creating the perceptron class

    def __init__(self):                 #initialize perceptron properties, synaptic weights  and the learning rate
        self.synaptic_weights = 2*np.random.random((5, 1)) - 1  #random weight values to start with
        self.learning_rate = 0.01       #I chose a small learning rate because it provided the best results
        self.bias = 1
        self.predictions = []
        self.actual_output = []

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
    
    def adjust(self, error, input):             #adjusts the weights of the synaptic weight
        lower = 1                           #if the guess was "lower" than the actual output then the error should be positive
        if error < 0:                       
            lower = -1                      #if the guess was higher than the actual output then the error should be negative
                            #I used the lower feature to manage the appropriate adjustments if y < D and if D < y
        for i in range(4):
            self.synaptic_weights[i] += lower*self.learning_rate*input[i]   #update all the weights
    
    def confusion_matrix_gen(self, preds, correct_output):
        
        setosa_matrix = np.array([0,0,0])      # [TP, FP, FP] for setosa
        versicolor_matrix  = np.array([0,0,0]) #[FP, TP, FP] for versicolor
        virginica_matrix = np.array([0,0,0])    #[FP, FP, TP] for virginica
        for i in range(len(preds)):                 #concept for this loop: if the correct_output == output then increase TP for that specific matrix
            if correct_output[i] == 0:          # if != then determine which is the other prediction and add to FP for that matrix
                if preds[i] == 0:
                    setosa_matrix[0] += 1
                elif preds [i] == 1:
                    setosa_matrix[1] += 1
                else:
                    setosa_matrix[2] += 1
            elif correct_output[i] == 1:
                if preds[i] == 0:
                    versicolor_matrix[0] += 1
                elif preds [i] == 1:
                    versicolor_matrix[1] += 1
                else:
                    versicolor_matrix[2] += 1
            else:
                if preds[i] == 0:
                    virginica_matrix[0] += 1
                elif preds [i] == 1:
                    virginica_matrix[1] += 1
                else:
                    virginica_matrix[2] += 1

        setosa_TN = versicolor_matrix[1] + virginica_matrix[2]
        versicolor_TN = setosa_matrix[0] + virginica_matrix[2]
        virginica_TN = setosa_matrix[0] + virginica_matrix[1]

        setosa_FP = setosa_matrix[1] + setosa_matrix[2]
        versicolor_FP = versicolor_matrix[0] + versicolor_matrix[2]
        virginica_FP = virginica_matrix[0] + virginica_matrix[1]

        setosa_FN = versicolor_matrix[0] + virginica_matrix[0]
        versicolor_FN = setosa_matrix[1] + virginica_matrix[1]
        virginica_FN = setosa_matrix[2] + versicolor_matrix[2]

        total_cases = 0
        for i in range(3):
            total_cases += setosa_matrix[i]
            total_cases += versicolor_matrix[i]
            total_cases += virginica_matrix[i]
        
        accuracy = setosa_matrix[0] + versicolor_matrix[1] + virginica_matrix[2]

        con_matrix =  [[setosa_matrix[0],setosa_matrix[1],setosa_matrix[2],setosa_matrix[0]/(setosa_FP + setosa_matrix[0])],
                       [versicolor_matrix[0],versicolor_matrix[1],versicolor_matrix[2],versicolor_matrix[1]/(versicolor_FP + versicolor_matrix[1])], 
                       [virginica_matrix[0],virginica_matrix[1],virginica_matrix[2],virginica_matrix[2]/(virginica_FP +  virginica_matrix[2])],
                       [setosa_matrix[0]/(setosa_FN + setosa_matrix[0]),versicolor_matrix[1]/(versicolor_FN + versicolor_matrix[1]),virginica_matrix[2]/(virginica_FN + virginica_matrix[2]), accuracy/total_cases]]

        return con_matrix

    def train(self, training_inputs, training_outputs, iterations): #train our data!
        
        for k in range(iterations):                         #iterate through the data multiple times for training
            for i in range(len(training_inputs)):           #run through each line of the training input individually
                output = float(np.dot(training_inputs[i], self.synaptic_weights))    #generate an output
                
                error = training_outputs[i] - output        #calculate the error
                self.adjust(error, training_inputs[i])      #adjust weights accordingly

                if output < 0.5:                    #if the output is less than 0.5, then I commit guess to being setosa
                    guess = 0
                elif 0.5 < output and output < 1.5: #if the output is greater than 0.5 but less than 1.5, then I commit guess to being versicolor
                    guess = 1
                else:
                    guess = 2 
                self.predictions.append(guess)
                self.actual_output.append(training_outputs[i])
        con_matrix = self.confusion_matrix_gen(self.predictions, self.actual_output)
        print("Training Matrix: ")
        for i in range(len(con_matrix)):
            print(con_matrix[i])
            
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
    perceptron.train(training_inputs, training_outputs, 100) #train the perceptron, runs through the data 100 times
    
    print("Synaptic weights post training: ")   #print finale weights for monitroing purposes
    print(perceptron.synaptic_weights)

    print("\nTime to Test\n")

    testing_inputs, testing_outputs = perceptron.organizeData("iris_test.txt") #load testing data

    correct = 0
    incorrect = 0
    predictions = []
    for i in range(len(testing_inputs)):                 #run through each line of the testing input and see if we guess correctly
        output = perceptron.think(testing_inputs[i])     #get output
        if output == testing_outputs[i]:
            correct += 1
        else:
            incorrect += 1
        predictions.append(output)

    confusion_matrix = perceptron.confusion_matrix_gen(testing_outputs, predictions)

    print("Correct: "+str(correct))
    print("Incorrct: "+str(incorrect))
    print("Test Confusion Matrix: ")
    for i in range(4):
        print(confusion_matrix[i])
    


