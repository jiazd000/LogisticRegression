
# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(X):  
    '''''Compute sigmoid function ''' 
    
    den =1.0+np.exp(-1.0*X)  
    gz =1.0/den  
    return gz 

def computeCost(X,y,theta):  
    '''''computes cost given predicted and actual values'''  
    m = X.shape[0]#number of training examples  

    J=0.0
    J=sigmoid(np.dot(X,theta)) 
    J=y*np.log(J)+(1.0-y)*np.log(1.0-J)

    J=J.sum()/(-m)
    

    return J

def gradientDescent(X, y, theta, alpha, iterations):  
    '''''compute gradient'''  
    m=X.shape[0]
    
    for i in range(iterations):
    # while(1):
        J=sigmoid(np.dot(X,theta))-y
        # print J
        # print DC(J,X[:,0]).sum()/m*alpha

        t1=theta[0]-(J*X[:,0]).sum()/m*alpha
        t2=theta[1]-(J*X[:,1]).sum()/m*alpha
        t3=theta[2]-(J*X[:,2]).sum()/m*alpha
        
        theta[0]=t1
        theta[1]=t2
        theta[2]=t3


    return theta

