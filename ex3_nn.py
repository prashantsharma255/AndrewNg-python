from ex3 import lrCostFunction , X, sigmoid,y
from scipy.io import loadmat
import numpy as  np

data = loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex3\ex3\ex3weights")
theta1 = data['Theta1'] 
theta2 = data['Theta2'] 

def predict(theta1, theta2, X):
    a1 = np.insert(X , [0] ,[1],1)
    a2 = sigmoid( a1 , theta1.T)
    a2 = np.insert(a2 , [0], [1] ,1)
    a3 = sigmoid(a2 , theta2.T)
    return np.argmax(a3,1)+1

pred = predict(theta1 , theta2 , X)
acc = (pred==y).mean()*100
print(acc)

