import numpy as np
from scipy.io import loadmat
import scipy.optimize as op

input_layer_size = 400
hidden_layer_size = 25
num_labels= 10

data= loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex4\ex4\ex4data1.mat",squeeze_me=True)
X = data['X']
y = data['y']
m = len(y)

data = loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex4\ex4\ex4weights")
theta1 = data['Theta1'] 
theta2 = data['Theta2'] 

#unrolling parameters
nn_params = theta1.reshape(theta1.size , 1)
nn_params= np.append(nn_params , [theta2.reshape(theta2.size , 1)])
nn_params = nn_params.reshape(len(nn_params) , 1)

#compute cost -feed forward
lamda = 3
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    t = 1/(1+np.exp(-z))
    return t*(1-t)
    
def randomInitialWeights(l_in , l_out):
    epsilon_init = 0.12
    W= np.random.rand(l_out , l_in+1)*2*epsilon_init -epsilon_init
    return W

initial_theta1 = randomInitialWeights(input_layer_size , hidden_layer_size)
initial_theta2 = randomInitialWeights(hidden_layer_size , num_labels)
initial_nn_params = np.concatenate((initial_theta1.reshape(initial_theta1.size, order='F'),\
                                    initial_theta2.reshape(initial_theta2.size, order='F')))

def nnCostFunction(nn_params , input_layer_size , num_labels , X , y , hidden_layer_size , lamda):
    
    #rolling theta vectors
    theta1 =nn_params[:hidden_layer_size*(input_layer_size+1)]  
    theta1 = theta1.reshape(hidden_layer_size , input_layer_size+1)
    theta2 = nn_params[hidden_layer_size*(input_layer_size+1):]
    theta2 = theta2.reshape(num_labels, hidden_layer_size+1)

    #initialization
    m =X.shape[0] 
    J=0
    
    #compute y[i] array  
    Y = np.zeros([m , num_labels])
    for i in range(m):
        Y[i,y[i]-1]= 1
    
    #Neural Network 
    a1 = np.insert(X, [0], [1],1)    #step 1
    z2 = a1.dot(theta1.T)
    a2 = np.insert(sigmoid(z2) , [0], [1] , 1)
    z3 = a2.dot(theta2.T)
    a3 = H = sigmoid(z3)

    #cost function 
    penalty = (lamda/(2*m))*( np.square(theta1[:,1:]).sum()+np.square(theta2[:,1:]).sum() )  
    J = (-1/m)*( Y*np.log(H) + (1-Y)*np.log(1-H) ).sum() +penalty 
    
    
    sigma3 = a3-Y    #step2
    sigma2 = (sigma3.dot(theta2)) * sigmoidGradient( np.insert(z2,[0],[1],1) )      #step 3
    sigma2 = sigma2[:,1:]       
    
    delta1 = (sigma2.T).dot(a1)   #step4
    delta2 = (sigma3.T).dot(a2)
    
    theta1_grad = delta1/m      #step 5
    theta1_grad[:,1:]+= (lamda/m)*theta1[:,1:]
    theta2_grad = delta2/m
    theta1_grad[:,1:]+= (lamda/m)*theta1[:,1:]
   
    #unroll gradients 
    grad = theta1_grad.reshape(theta1_grad.size , 1)
    grad = np.append(grad , [theta2_grad.reshape(theta2_grad.size , 1)])
    grad = grad.reshape(len(grad) , 1)
    
    return J  ,grad
    
#J , grad = nnCostFunction(nn_params , input_layer_size , num_labels , X , y , hidden_layer_size , lamda)

maxiter = 400
myargs= (input_layer_size , num_labels , X , y , hidden_layer_size, lamda)
result = op.minimize(nnCostFunction , x0= initial_nn_params ,args= myargs ,\
                     options={'disp': True, 'maxiter':maxiter} ,method = "TNC",jac=True)

nn_params=result.x
theta1 =nn_params[:hidden_layer_size*(input_layer_size+1)]  
theta1 = theta1.reshape(hidden_layer_size , input_layer_size+1)
theta2 = nn_params[hidden_layer_size*(input_layer_size+1):]
theta2 = theta2.reshape(num_labels, hidden_layer_size+1)


def predict(theta1, theta2, X):
    a1 = np.insert(X , [0] ,[1],1)
    z2=a1.dot(theta1.T)
    a2 = sigmoid( z2)
    a2 = np.insert(a2 , [0], [1] ,1)
    z3=a2.dot(theta2.T)
    a3 = sigmoid(z3)
    return np.argmax(a3,1)+1

pred = predict(theta1 , theta2 , X)
acc = (pred==y).mean()*100
print(acc)


'''
def checkNNGradients(lamda):
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;

    theta1 =  np.zeros([hidden_layer_size , input_layer_size+1])
    theta1 = np.reshape(np.sin(range(theta1.size)) ,theta1.shape )/10
    theta2 = np.zeros([num_labels , hidden_layer_size+1])
    theta2 = np.reshape(np.sin(range(theta2.size)) ,theta2.shape )/10
    X = np.zeros([m , input_layer_size])
    X = np.reshape(np.sin(range(X.size)) ,X.shape )/10
    y = 1+np.mod(range(m) , num_labels).T
    
lamda = 3
checkNNGradients(lamda)
'''






