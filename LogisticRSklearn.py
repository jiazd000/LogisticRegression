# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.font_manager import FontProperties 
from sklearn.linear_model import LogisticRegression

data=np.loadtxt('ex2data1.txt')

X=data[:,0:2]

y=data[:,2]

m=X.shape[0]

model=LogisticRegression()

model.fit(X,y)

SS=model.score(X,y)
print 'score=' '%.2f' % SS 
plt.figure(figsize=(8,6))
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 第0列的范围
x2_min, x2_max = X[:, 1].min(), X[:, 1].max() 
print x1_max

x1,x2=np.mgrid[x1_min:x1_max:200j,x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
print grid_test

grid_test_pre=model.predict(grid_test)

grid_test_pre=grid_test_pre.reshape(x1.shape)

print grid_test_pre
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])   
plt.pcolormesh(x1,x2,grid_test_pre,cmap=cm_light)

pos = np.where(y == 1)  
neg = np.where(y == 0)  

myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X[pos,0],X[pos,1],color="red",label="Admitted",linewidth=3) 
plt.scatter(X[neg,0],X[neg,1],color="blue",label="Not admitted",linewidth=3) 
plt.xlabel(u'Exam1 Score',fontproperties=myfont)
plt.ylabel('Exam2 Score')

plt.legend()
plt.show()


