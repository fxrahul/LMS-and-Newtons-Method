# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:02:04 2019

@author: 91755
"""
import numpy as np 
import pandas as pd

import numdifftools as nd



class NewtonMethod():
    
    def __init__(self):
       np.random.seed(1)
       self.weights =  np.random.random((2,1)) 

    def activation(self,x):
       
       calculate = 1/(1+np.exp(-x.astype(float)))
       calculate[calculate >= 0.7] = 1
       calculate[calculate < 0.7] = -1
       return calculate


    
   
    
    def training_time(self, train_inputs, train_outputs, no_of_epoch):
        
        
        
        
        #a = [0.1,0.01,0.001]
  
        for i in range(no_of_epoch):
            #l = (int) (i / 17)
            learning_rate = 0.01
            pred_output = self.predValue(train_inputs)
            error = train_outputs - pred_output
            
            mse = np.square(np.subtract(train_outputs,pred_output)).mean()
        
            
            func = lambda x: mse*x[0]**2 + mse*x[1]**3 + x[0]*x[1] 
            w = self.weights
            w1 = w[0][0]
            w2 = w[1][0]
            w_array = [w1,w2]
#            grad = gradient(w_array, func, eps=1e-4)
#            hess = hessian(w_array, func, eps=1e-4)
            nd_hess = nd.Hessian(func)
            nd_hess_vals = nd_hess(w_array)

            
            gradient_matrix = (1/train_inputs.size) *(np.dot(train_inputs.T, error))
           

          
            
          
            hessian_matrix = nd_hess_vals
            #gradient_vector = np.array([] [])
            
            hessian_inverse = np.linalg.inv(hessian_matrix)
            ajustmentForWeights = learning_rate  * np.dot(hessian_inverse,gradient_matrix)
            self.weights = self.weights -  ajustmentForWeights
       

        #print(np.amax(pred_output))
        
            
            
    def predValue(self,inputs):
        np.set_printoptions(suppress=True) #prevents exponential value
        inputs = inputs.astype(float)
    
        output = np.dot(inputs, self.weights)
        o = self.activation(output)
#        threshold = (np.amax(output) + np.amin(output)) / 2
#        output[ output >= threshold ] = 1
#        output[ output < threshold ] = -1

 
        return o
    
   
    
    def predAccuracy(self,originalLabel,predValue):

        matched = 0
        for i in range(len(originalLabel)):
                if originalLabel[i] == predValue[i]:
                    matched += 1
        accuracyVal = matched / float(len(originalLabel)) * 100.0       
        return accuracyVal
    
    
    
   
    
    
    
if __name__ == "__main__":
           
    data = pd.read_csv("cc.csv") 
        # Preview the first 5 lines of the loaded data 
    data.head()
    inputData = pd.DataFrame(data, columns = ['a7', 'a10']).to_numpy()
    outLabelWithoutBinary = pd.DataFrame(data, columns = ['a15']).to_numpy()
    outLabelWithoutBinary[outLabelWithoutBinary == '+'] = 1
    outLabelWithoutBinary[outLabelWithoutBinary == '-'] = -1
    
    row, col = outLabelWithoutBinary.shape
    no_of_train_data = (int) (row/2)
    no_of_test_data = row - no_of_train_data
    #train data
    inputTrainData = inputData[0 : no_of_train_data,:]
    outputTrainLabel = outLabelWithoutBinary[0 : no_of_train_data,:]
    
    nn = NewtonMethod()
    
    
    nn.training_time(inputTrainData,outputTrainLabel,50)
    
    #test data
    inputTestData = inputData[no_of_test_data: row,:]
    outputTestLabel = outLabelWithoutBinary[no_of_test_data : row,:]
    
    
        

    
    uciOutput = nn.predValue(inputTestData)
    accuracy = nn.predAccuracy(outputTestLabel,uciOutput)
     
    print("UCI Dataset Accuracy using Newton's Method :",accuracy, "%")