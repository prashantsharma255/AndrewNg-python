import pandas as pd
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import scipy.optimize as op

data = loadtxt(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex2\ex2\ex2data1.txt", delimiter=',')

X= data[:,0:2]
y= data[:,2]
y= y.reshape(len(y),1)

#plotting
def plotData(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)

    plt.scatter(X[pos , 0] , X[pos , 1]  , marker = 'x' ,c='b', label="Admitted")
    plt.scatter(X[neg , 0] , X[neg , 1] ,marker = 'o' , c='r' ,label="Not admitted")
    plt.legend()
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")    
    
plotData(X,y)

#Computing cost function and gradient descent
X = np.insert( X , [0], [1], 1)
initial_theta = np.zeros([len(X[0]), 1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def computeCost(initial_theta, X, y):
    m = len(y)
    h = sigmoid( np.dot(X, initial_theta) )
    h=h.reshape(len(h),1)
    #term1 = y*np.log(h)
    #term2 = (1-y)*np.log(1-h)
    #J = (-1/m)* ( (term1 + term2).sum() )
    J = (-1/m)* ( (y.T).dot(np.log(h)) + ((1-y).T).dot(np.log(1-h)) )  #upper three lines of code club in to one 
    grad = (1/m)*( (X.T).dot(h-y) )
    return J,grad

''' 
def gradient(initial_theta, X , y):    #Not necessary to use this function 
    m = len(y)
    h = sigmoid(X.dot(initial_theta))
    #term = (1/m)*( (h-y)*X ).sum(0)
    #term = term.reshape(len(term),1)
    term = (1/m)*( (X.T).dot(h-y) )    
    print(term.shape) 
    return term
'''

#without writing Gradient func explicitly  
cost , grad = computeCost(initial_theta, X, y)
print("Cost of initial=",cost)
print("Gradient of initial=",grad) 

test_theta= np.array([[-24],[0.2],[0.2]]) 
cost,grad = computeCost(test_theta, X, y)
print("\nCost of test=",cost)
print("Gradient of test=",grad )

#Optimizing 
result = op.minimize(fun = computeCost , x0= initial_theta ,args=(X,y), method='TNC',jac=True)
print("\nCost at theta found by op.min is ",result.fun)
print("Theta found by op.min ",result.x)
theta = (result.x).reshape(len(result.x),1)

#Plot Boundary
def plotDecisionBundary(theta, X, y):
    if( len(X[0]) <= 3):
        plotX= np.array([ [X[:,1].min()-2] , [X[:,1].max()+2]  ])
        plotY = np.array([ (-1/theta[2])*( theta[1]*plotX +theta[0] )  ])
        plotY= plotY.reshape(len(plotX), 1)
        plt.plot(plotX,plotY, label="Boundary", c='g')
        plt.legend()
        plt.axis([30,100,30,100])
    else:
        u = np.linspace(-1 , 1.5 , 50)
        v = np.linspace(-1 , 1.5 , 50)
        z = np.array([[len(u)] , [len(v)] ])
        for i in range(1, len(u)):
            for j in range(1, len(v)):
                z[i, j] = (mapFeature( np.array([ u[i] ]),np.array([ v[j]]) ).dot(np.array(theta)))
    z = z.T
    plt.contour(u, v, z)
        
plotDecisionBundary(theta, X , y)

#Predict
probs = sigmoid( np.array([ [1,45,85] ]).dot(theta) )
print("\nFor a student with scores 45 and 85, we predict an admission probability of" , probs)

#Accuracy
def predict(theta , X):
    m = len(X)
    p = np.zeros( [m ,1])
    for i in range( 0 , m):
        if sigmoid( X[i].dot(theta) )>=0.5:
            p[i]=1
        else:
            p[i]=0
    return p
    
p = predict(theta , X)
acc = ((p==y).mean())*100
print("\nTrain accuracy: ",acc)

