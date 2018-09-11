# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 18:35:26 2018

@author: Hp
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import imageio
import random as rn

data = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex7\ex7\ex7data2.mat')
X = data['X']

#K = 3       #no. of clusters

initial_centroids = np.array([[3,3],[6,2],[8,5]])

def findClosestCentroids(X, initial_centroids):
    idx = np.zeros([len(X),1])
    m = len(X)
    
    for i in range(m):
       x= ((X[i]-initial_centroids)**2).sum(axis = 1) 
       idx[i] =int( np.argmin(x) )
    
    return idx

def computeCentroids(X , idx , K):
    m = len(X)
    col_len = len(X[0])
    x2 = np.array([1, col_len])
    centroids= np.zeros([K,col_len])
    
    for k in range(K):   
        x1 = [i for i in range(m) if idx[i]==k ]
        len_of_k = len(x1)
    
        x2 = X[x1].sum(0)
        centroids[k] = x2/len_of_k
        
        x1.clear()
        x2 = 0
        
    return centroids

#idx = findClosestCentroids(X , initial_centroids)
#centroids = computeCentroids(X ,idx ,K)

def runkMeans(X , initial_centroids , max_iters, plot_bool ,K):
    if plot_bool==True:
        centroids_for_plot = np.zeros([max_iters+1 , 2*K])    
        centroids_for_plot[0,:] = initial_centroids.ravel()   
    
        plt.scatter(X[:,0] , X[:,1] )
        plt.scatter( initial_centroids[:,0],initial_centroids[:,1], marker='x',c='black' )
    
    centroids =initial_centroids
    for i in range(max_iters):
        idx = findClosestCentroids(X , centroids)
        centroids = computeCentroids(X , idx ,K)
        if plot_bool==True:
            centroids_for_plot[i+1,:]=centroids.ravel()
        
    if plot_bool == True:
        centroidsPlotting(X , centroids_for_plot ,K,idx)
        return centroids , idx , centroids_for_plot
    else:
        return centroids , idx

def centroidsPlotting(X , centroids_for_plot , K,idx ):
    for l in range(K):         #tracing iteration of centroids
        m,n = 2*l,2*l+1
        plt.plot(centroids_for_plot[:,m],centroids_for_plot[:,n], marker ='x')
    
    for i in range(K):    #plotting final clusters
        pos,neg = np.where(idx==i)
        plt.scatter(X[pos,0],X[pos,1],marker ='o')

#centroids , idx , centroids_for_plot = runkMeans(X , initial_centroids ,10 , True)    



#kmeans clustering on pixels
A = imageio.imread(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex7\ex7\bird_small.png')
A = A/A.max()

img_size = A.shape
X2 = A.reshape([img_size[0]*img_size[1],3])

K2 = 16
max_iters = 10

def kMeanInitCentroids(X , K):
    centroids = np.zeros([K , len(X[0])])
    
    randidx = np.zeros([len(X),1])
    randidx = [ rn.randrange(0,len(X)) for i in range(len(X))]
    
    for i in range(K):
        centroids[i,:] = X[randidx[i] , :]
      
    return centroids
    
initial_centroids2 = kMeanInitCentroids(X2, K2)

centroids2 , idx2 = runkMeans(X2 ,initial_centroids2 , max_iters , False ,K2)

idx2 = findClosestCentroids(X2 , initial_centroids2)
idx2 = [ int(x) for x in idx2]
X_recovered = centroids2[idx2,:]
X_recovered = X_recovered.reshape([img_size[0],img_size[1],3])
 
#plot A and X_recovered using imagesc like function to obtain original and compressed image

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(A)
ax1.set_title('original')
ax2.imshow(X_recovered)
ax2.set_title('compressed')

    