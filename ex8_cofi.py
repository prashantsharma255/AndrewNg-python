# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:50:06 2018

@author: Hp
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as op

data = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex8\ex8\ex8_movies.mat')
Y = data['Y']
R = data['R']

params = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex8\ex8\ex8_movieParams.mat')
X = params['X']
Theta = params['Theta']
num_users = params['num_users']
num_movies = params['num_movies']
num_features = params['num_features']


'''num_users = 4           #for faster calculation purpose
num_movies = 5
num_features = 3
X = X[:num_movies,:num_features]
Theta = Theta[:num_users , :num_features]
Y = Y[:num_movies,:num_users]
R = R[:num_movies,:num_users]
'''
def cofiCostFunc(params,Y,R ,num_features, Lambda):
    
    #Y = np.matrix(Y)  # (1682, 943)
    #R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    J = 0
    #X = X.reshape([1682,10])
    diff = np.dot(X,Theta.T)
    diff = np.multiply(diff ,R)
    J =0.5*(np.power((diff - Y),2)).sum() + (Lambda/2)*(np.power(Theta,2)).sum() + \
    (Lambda/2)*(np.power(X,2)).sum()
    
    X_Grad = np.dot((diff - Y),Theta) + Lambda*X   
    Theta_Grad = np.dot((diff-Y).T,X) + Lambda*Theta
    
    grad = np.concatenate((np.ravel(X_Grad), np.ravel(Theta_Grad)))
    return J , grad

def normalizeRatings(Y,R):
    Ymean = np.zeros([len(Y),1])
    m,n = Y.shape
    Ynorm = np.zeros([m,n])
    for i in range(len(Y)):
        idx = np.where(R[i,:]==1)[0]
        Ymean[i] = np.mean(Y[i,idx])   
        Ynorm[i,idx] = Y[i,idx]-Ymean[i,0]
        
    return Ymean , Ynorm



data2 = open(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex8\ex8\movie_ids.txt').readlines()
movieList = []

for i in data2:
    a,b = i.split(' ',1)
    movieList.append(b)
movieList = [i.lstrip() for i in movieList]
movieList = [i.rstrip('\n') for i in movieList]

my_ratings = np.zeros([1682,1])
my_ratings[0] = 4
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[97] = 2
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

#inserting new values
Y = np.append(Y,my_ratings,1)
idx = np.where(my_ratings!=0)[0]

'''print("Original ratings")
for i in idx:
    print("Orignal rating-{0},Movie- {1}".format( my_ratings[i,0], movieList[int(i)] ))
'''
for i in idx:   #updating R of new ratings 
    my_ratings[i] = 1 
R = np.append(R,my_ratings,1)

Ymean , Ynorm = normalizeRatings(Y,R)    # normalizing ratings 
#print(Ynorm[idx,-1]+Ymean[idx,0])

num_users = len(Y[0])    #useful values
num_movies = len(Y)
num_features = 10

X_init = np.random.random([num_movies, num_features])   #initializing to random values
Theta_init = np.random.random([num_users,num_features])

params = np.concatenate((np.ravel(X_init), np.ravel(Theta_init)))

a = op.minimize(fun=cofiCostFunc , x0=params ,args=( Y ,R ,10,10)\
                ,method='CG',jac=True,options={'maxiter':50 })

#unrolling values for trained model
X = np.array(np.reshape( a.x[:num_movies*num_features],(num_movies,num_features) ))
Theta = np.array(np.reshape( a.x[num_movies*num_features:],(num_users,num_features) ))

#predicting ratings for our ratings
p = np.dot(X,Theta.T)
my_pred = p[:,-1].reshape([-1,1]) + Ymean

idx = np.argsort(my_pred,0)[::-1]
idx = idx[:10]
print("\nPredicted ratings")
for i in idx:
    print("Rating-{0}, {1}".format( '%0.2f'%my_pred[int(i)], movieList[int(i)] ))
print(a)
