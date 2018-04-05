import numpy as np
import scipy.optimize as op
from numpy import loadtxt
import matplotlib.pyplot as plt

data =  loadtxt(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex2\ex2\ex2data2.txt", delimiter=',')
X= data[:,[0,1]]
y= data[:,2].reshape(len(data),1)

#plotting
def plotData(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.scatter(X[pos , 0] , X[pos , 1]  , marker = 'x' ,c='b')
    plt.scatter(X[neg , 0] , X[neg , 1] ,marker = 'o' , c='r' )
    plt.legend(['y=1','y=0'])
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")    
    
plotData(X,y)

#Regularization
def mapFeature(X1 , X2):
    degree = 6
    X1 = X1.reshape(len(X1),1)
    X2 = X2.reshape(len(X2),1)
    out = np.ones([len(X1) ,1])
    for i in range(1, degree+1 ):
        for j in range(i+1):
            r = (X1 ** (i - j)) * (X2 ** j)
            out = np.append(out, r, axis=1)
    return out

X = mapFeature( X[:,0], X[:,1])
initial_theta = np.zeros([len(X[0]), 1])
lamda = 1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionReg(initial_theta, X, y, lamda):
    m = len(y)
    h = sigmoid( np.dot(X, initial_theta) )
    h=h.reshape(len(h),1)
    grad = np.zeros([len(initial_theta),1])
    
    initial_theta = initial_theta.reshape(len(initial_theta),1)
    lamdaTerm=lamda/(2*m)*np.square(initial_theta).sum()
    J = (-1/m)* ( (y.T).dot(np.log(h)) + ((1-y).T).dot(np.log(1-h)) ) + lamdaTerm  
    
    grad[0]     = 1/m*( (X[:,0].T).dot(h-y) )
    grad[1:]     = 1/m*( (X[:,1:].T).dot(h-y) ) + (3/m)*initial_theta[1:]

    return J,grad
       
[cost, grad] = costFunctionReg(initial_theta, X, y, lamda)
print("\nCost at initial theta (zeros):" , cost)
print("Gradient at   initial theta (zeros)", grad[:5])

#Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones([len(X[0]),1])
cost, grad = costFunctionReg(test_theta, X, y,10 )
print("\nCost at initial theta (ones):" , cost)
print("Gradient at   initial theta (ones)", grad[:5])

#Optimizing 
lamda = 1
result = op.minimize(fun = costFunctionReg , x0= initial_theta ,args=(X,y, lamda), method='TNC',jac=True )                    
print("\nCost at theta found by op.min is ",result.fun)
print("Theta found by op.min ",result.x[:5])
theta = (result.x).reshape(len(result.x),1)

#plot boundary
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
        z = np.zeros([len(u), len(v)])
        for i in range(1, len(u)):
            for j in range(1, len(v)):
                z[i, j] = (mapFeature( np.array([ u[i] ]),np.array([ v[j]]) ).dot(np.array(theta)))
    z = z.T
    plt.contour(u, v, z )
    plt.title('lambda = %f' % lamda)
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    
plotDecisionBundary(theta, X, y)        

#predict accuracy
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

 
    
            



