import numpy as np
import matplotlib.pyplot as plt

f = open(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex1\ex1\ex1data2.txt",'r')
x,x1, y = zip(*[ i.split(',') for i in f.readlines()] )

x = [float(i) for i in x]
y= [float(i) for i in y]
x1 = [float(i) for i in x1]

x = np.asarray(x).reshape([len(x),1])
x1 = np.asarray(x1).reshape([len(x1) , 1])
y = np.asarray(y).reshape([len(x),1])

x = np.insert( x , [1] ,x1 , 1)

m = len(y)

#part1 scale features and set them to zero mean
def featureNormalize(x):
    xNorm = x
    mu = np.zeros([ 1, len(x[0])])
    sigma = np.zeros([1, len(x[0])])
    
    mu = [ x[:,i].mean() for i in range(len(x[0]))]
    sigma = [ x[:,i].std() for i in range(len(x[0]))]
    
    xMinusMu = x - mu
    xNorm = xMinusMu/sigma
    
    return (xNorm , mu , sigma)
    
[x , mu ,sigma] = featureNormalize(x)

x = np.insert(x , [0] ,[1] , 1)

#part2 gradient descent
def computeCost(x, y, theta):
    m = len(y)
    pred = np.dot(x ,theta)
    diff = np.square(pred - y)
    cost = 1/(2*m)*(diff.sum())    
    return cost

alpha = 0.1
num_iters = 50
theta = np.zeros([len(x[0]),1])

def gradientDescentMulti( x, y , theta , alpha , num_iters ):
    m = len(y)
    J_history = np.zeros([num_iters,1])
    theta_history = np.zeros([num_iters ,len(x[0])])
    
    for i in range(num_iters):
        
        #h = theta[0]*x[:,0] + theta[1]*x[:,1] +theta[2]*x[:,2]    #this is one way of doing 
        #h = h.reshape(len(h),1)
        
        #this is a generalised way of solving this problem
        h = (np.transpose(theta))*x
        h = np.sum(h, axis= 1)
        h = h.reshape(len(x),1)
        
        term = ((h-y)*x).sum(axis=0)
        term = term.reshape(len(term),1)
        
        theta = theta - alpha * (1/m) * term
        
        theta_history[i]=[theta[0] , theta[1] , theta[2]]
        
        J_history[i] = computeCost(x, y , theta)
        
    minVal = J_history.min()
  
    for j in range(len(J_history)):
        if J_history[j]==minVal:
            index=j
            break
 
    return (theta_history[index], J_history)

[theta , J_history] = gradientDescentMulti(x , y , theta , alpha , num_iters)
theta =theta.reshape(len(theta),1)

#plotting convergence graph
plt.plot( J_history, color='g' )
plt.xlabel("No. of iterations")
plt.ylabel("Cost J")
plt.title("Convergence graph for J(theta)")

# Display gradient descent's result
print('Theta computed from gradient descent: ');
print(theta);

#Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot( np.array([ [1 , (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]] ]),theta)
print("\nPredicted price of a 1650 sq-ft, 3 br house (using gradient descent)is :" ,float(price) )


