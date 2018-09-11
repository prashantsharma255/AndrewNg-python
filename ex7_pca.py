# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:10:33 2018

@author: Hp
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np


data = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex7\ex7\ex7data1.mat')
X = data['X']

def featureNormalize(X):
    X_norm = X
    
    mu = X.mean(0)
    XMinusMu = X - mu
    sigma = XMinusMu.std(0)
    X_norm = XMinusMu/sigma
    
    return X_norm, mu ,sigma

X_norm , mu , sigma = featureNormalize(X)
    
def pca(X):
    m = len(X)
    
    sigma =(1/m)*np.dot(X.T,X)
    U , S ,V = np.linalg.svd(sigma)
    
    return U , S

U , S = pca(X_norm)    

def draw_line(p1,p2):
    plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), c='r', linewidth=1)
    print(p1[0],p2[0],p1[1], p2[1])
    
'''draw_line(mu, mu + 1.5 * S[0] * U[:, 0]) 
draw_line(mu , mu + 1.5 * S[1] * U[:,1] )
plt.scatter(X[:,0] , X[:,1] , marker='o' )
'''   

def projectData(X , U , K):
    
   Ureduce = U[:,:K]
   Z = np.dot(X , Ureduce)
   return Z
    
def recoverData(Z , U , K):
    X_rec = np.dot(Z, U[:,:K].T)
    return X_rec
    
'''K=1
Z = projectData(X_norm , U ,K)
X_rec = recoverData(Z, U,K)

plt.scatter(X_rec[:,0]  , X_rec[:,1] , c='r')
plt.scatter(X_norm[:,0] , X_norm[:,1])
'''

data2 = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex7\ex7\ex7faces.mat')
X2 = data2['X']

def displayData(X):
    m ,n = X.shape
    
    example_width = int( np.round(np.sqrt(n)) )
    example_height = int(n/example_width)
    display_rows = int( np.floor(np.sqrt(m)) )
    display_cols = int( np.ceil(m/display_rows))
    
    pad = 1
    displayArray = -np.ones( [pad + display_rows*(example_width+pad),\
                              pad + display_cols*(example_height+pad)] )
    curr_ex = 1    #current example count
    
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex>m:
                break
            
            rows = pad + i*(example_height+pad) + np.array(range(example_height))
            cols = pad + j*(example_width+pad) + np.array(range(example_width))
            
            displayArray[ rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1 ] = np.reshape( X[curr_ex-1,:] ,\
                        (example_height,example_width ) , order='F')

            curr_ex = curr_ex+1
       
        if curr_ex>m :
            break
        
    plt.imshow(displayArray)
    plt.axis('off')
    
#displayData(X2[:100,:])

'''X_norm2 , mu2 , sigma2 = featureNormalize(X2)
U2 , S2 = pca(X_norm2)
displayData(U2[:,:36].T)

K2=100
Z2 = projectData(X2 , U2 ,K2) 
X_rec = recoverData(Z2 , U2 ,K2)

print('Original')
displayData(X_norm2[:100,:])
print('Recovered')
displayData(X_rec[:100,:])
'''