# -*- coding: utf-8 -*-
from LogisticR import sigmoid
import numpy as np
import matplotlib.pyplot as plt 

theta=np.array([[1.0,1.0],[2.3,2.2]])

data=np.loadtxt('aa.txt')

X=data[:,0]

y=np.array([1,2,3])
# print theta.shape
# print theta[1][0]
# print X,np.dot(X,theta).shape
# print X.shape
# print np.dot(X,theta)
# # print sigmoid(a)
# print X[:,0]*X[:,0]

# t=np.array([1.0,2.0,3.0])
# h=np.array([1.0,2.0,3.0])
# print t.shape,h.shape
# print t*h
x=np.linspace(-20,20,100)
y=sigmoid(x)
plt.plot(x,y,color="orange",label="Fitting Line",linewidth=2) 
plt.legend()
plt.show()
print 1/(np.exp(X)+1)
