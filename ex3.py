import pandas as pd
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as op

input_layer_size  = 400
num_labels = 10

data= loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex3\ex3\ex3data1.mat",squeeze_me=True)

X = data['X']
y = data['y']
m = len(y)

rand_int = random.sample(range(0,m) , 101)
sel = X[rand_int[1:101], :]

def displayData(X):
    
    example_width = int(np.round( np.sqrt( len(X[0] ) ) ))
    
    plt.figure()
    plt.set_cmap("gray")
    
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))
    
    m , n = X.shape
    example_height =int( n/example_width)
    
    display_row =int( np.floor( np.sqrt(m) ))
    display_col =int( np.ceil( m/display_row ))
    
    pad = 1
    display_array = - np.ones([ pad+display_row*(example_height+pad) , pad+display_col*(example_width+pad) ])
    curr_ex = 1
    for j in range(1 , display_row+1):
        for i in range(1 , display_col+1):
            if curr_ex>m:
                break
            
            max_val =np.max( np.abs( X[curr_ex-1 ,:] ))
            
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))
           
            display_array[ rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape( X[curr_ex-1, :], (example_height, example_width) , order='F') #/ max_val
            curr_ex = curr_ex +1
           
        if curr_ex>m:
            break
        
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    plt.axis('off')
    return h, display_array

#displayData(sel)

#Cost function with regularization
theta_t = np.array([ [-2],[-1],[1],[2] ])
a=np.arange(1,16).reshape((5, 3), order='F')/10
X_t= np.insert( np.ones([5,1]) , [1], a, 1)
y_t =np.array([ [1],[0],[1],[0],[1] ])  
lambda_t = 3

def sigmoid(X , theta) :
    return 1/(1+ np.exp(-(X.dot(theta)) ))

def lrCostFunction(theta, X , y , l):
    m =len(y)
    J= 0
    grad = np.zeros([len(theta) , 1])
    
    h = sigmoid( X, theta )
    lambda_term = l/(2*m)*np.square(theta[1:]).sum() 
    J = (-1/m)*( ( y*np.log(h) + (1-y)*np.log(1-h) ).sum() )  + lambda_term

    l_grad = (l/m)*theta
    l_grad[0]=0
   
    grad = (1/m)*( (X.T).dot(h-y) ) +l_grad
    

    return J , grad

'''  
J ,grad =lrCostFunction(theta_t , X_t , y_t, lambda_t)
print("Cost of test case",J)
print("Gradients of test case", grad)
'''

#one vs all
l= 0.1

def oneVsAll(X, y , num_labels, l):
    m, n  = X.shape
    all_theta = np.ones([num_labels ,n+1])   ##
    X = np.insert(X , [0],[1],1)
    theta = np.zeros([n+1,1])
    
    for c in range(1, num_labels+1):
        y_changed = np.where(y==c, 1,0)
        a= op.minimize(fun = lrCostFunction , x0=  theta , args=(X, y_changed , l) , method = 'CG' , jac=True)
        all_theta[c-1,:]=a.x
    
    return all_theta
    
all_theta = oneVsAll(X, y , num_labels , l)

#predict for one vs all
def predictOneVsAll(all_theta, X):
    X = np.insert(X , [0],[1],1)
    return np.argmax(sigmoid(X , all_theta.T), axis=1)+1
    
p = predictOneVsAll(all_theta , X)
acc = (p==y).mean()*100
print(acc)


