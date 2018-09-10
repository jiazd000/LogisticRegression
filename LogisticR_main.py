# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties 
from LogisticR import computeCost,gradientDescent

data=np.loadtxt('ex2data1.txt')

X=data[:,0:2]

y=data[:,2]

m=X.shape[0]

X=np.column_stack((np.ones((m,1)),X))

theta=np.array([0.0,0.0,0.0])

iterations = 200000
alpha = 0.01

theta = gradientDescent(X, y, theta, alpha, iterations)
J=computeCost(X, y, theta)
print theta,J
pos = np.where(y == 1)  
neg = np.where(y == 0)  
plt.figure(figsize=(8,6))
myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X[pos,1],X[pos,2],color="red",label="Admitted",linewidth=3) 
plt.scatter(X[neg,1],X[neg,2],color="blue",label="Not admitted",linewidth=3) 
plt.xlabel(u'Exam1 Score',fontproperties=myfont)
plt.ylabel('Exam2 Score')
x1=np.linspace(20,110,100)
x2=-(theta[1]*x1+theta[0])/theta[2]
plt.plot(x1,x2,color="orange",label="Fitting Line",linewidth=2) 
plt.legend()
plt.show()
