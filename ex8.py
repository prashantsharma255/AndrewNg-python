# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:58:20 2018

@author: Hp
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

data = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex8\ex8\ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

def estimateGaussian(X):
    m ,n = X.shape
    mu = np.mean(X,0)#.reshape([n,1])
    sigma2 = np.var(X, 0)#.reshape([n,1])
    return mu, sigma2
    
def multivariateGaussian(mu , sigma2 , X):
    
    k = len(mu)
    sigma2 = np.diag(sigma2)
    
    X = X - mu
    p = ( (2*np.pi)**(-k/2) * np.linalg.det(sigma2)**(-0.5)) *\
    np.exp(-0.5 *np.sum( np.dot(X ,np.linalg.inv(sigma2)) * X , 1))
    
    return p.reshape([-1,1])

def visualizeFit(X , mu , sigma2):      #not working
    
    meshvals = np.arange(0, 35, .5)
    X1, X2 = np.meshgrid(meshvals, meshvals)
    Z = np.hstack((X1.reshape((-1,1)), X2.reshape((-1,1))))
    Z = multivariateGaussian(Z, mu, sigma2).reshape(np.shape(X1))

    mylevels = [10**exp for exp in range(-20,0,3)]
    mylevels = np.array(mylevels).reshape([-1,1])
    plt.contour(X1, X2, Z, levels=mylevels)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.xlabel('Latency(ms)')
    plt.ylabel('Throughput(mb/s)')
    
def selectThreshold(yval , pval):
    step = ( np.max(pval) - np.min(pval) )/1000
    epsilon = np.min(pval) + step
    ypred = np.zeros([len(yval),1])
    f1_best = 0 
    epsilon_best = 0
    
    while epsilon < np.max(pval):
        
        ypred = (pval<epsilon).reshape((-1,1))
        tp = np.sum( (yval==1) & (ypred==1) )
        fp = np.sum( (yval==0) & (ypred==1) )
        fn = np.sum( (yval==1) & (ypred==0) )
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec)
        
        if f1_best<f1:
            f1_best = f1
            epsilon_best = epsilon
            
        epsilon = epsilon+step
    return epsilon_best,f1_best
    
mu , sigma2 = estimateGaussian(X)
p = multivariateGaussian(mu , sigma2 , X)
#visualizeFit(X , mu, sigma2)                        #not working

pval = multivariateGaussian(mu , sigma2, Xval)
epsilon , f1 = selectThreshold(yval,pval)

outliers = (p<epsilon).reshape([-1,1])
i = np.where(outliers==1)
i = i[0]
#plt.scatter(X[i,0] , X[i,1],marker='x', c='red')


#Multidimensional outliers
data2 = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex8\ex8\ex8data2.mat')
X2 = data2['X']
Xval2 = data2['Xval']
yval2 = data2['yval']
mu2 , sigma2_2 = estimateGaussian(X2)
p2 = multivariateGaussian(mu2 , sigma2_2 ,X2)

pval2 = multivariateGaussian(mu2,sigma2_2,Xval2)
epsilon2 , f12 = selectThreshold(yval2 , pval2)

#print(epsilon2,f12)
#print(np.sum(p2<epsilon2))
