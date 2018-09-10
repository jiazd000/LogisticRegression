def sigmoid(X):  
    '''''Compute sigmoid function '''  
    den =1.0+ e **(-1.0* X)  
    gz =1.0/ den  
    return gz  
def compute_cost(theta,X,y):  
    '''''computes cost given predicted and actual values'''  
    m = X.shape[0]#number of training examples  
    theta = reshape(theta,(len(theta),1))  
      
    J =(1./m)*(-transpose(y).dot(log(sigmoid(X.dot(theta))))- transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))  
      
    grad = transpose((1./m)*transpose(sigmoid(X.dot(theta))- y).dot(X))  
    #optimize.fmin expects a single value, so cannot return grad  
    return J[0][0]#,grad  
def compute_grad(theta, X, y):  
    '''''compute gradient'''  
    theta.shape =(1,3)  
    grad = zeros(3)  
    h = sigmoid(X.dot(theta.T))  
    delta = h - y  
    l = grad.size  
    for i in range(l):  
        sumdelta = delta.T.dot(X[:, i])  
        grad[i]=(1.0/ m)* sumdelta *-1  
    theta.shape =(3,)  
    return  grad 