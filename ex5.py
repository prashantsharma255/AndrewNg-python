from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

data = loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex5\ex5\ex5data1.mat")

X = data["X"]
y = data["y"]
Xval = data["Xval"]
yval = data["yval"]
Xtest = data["Xtest"]
ytest = data["ytest"]
m = len(y)

#regularized linear regression
theta = np.ones([X.shape[1]+1,1])
l=1
def linearRegCostFunction(theta,X, y, l):
    m = len(y)
    J= 0
    grad = np.zeros([len(theta) , 1])
    theta = theta.reshape(len(theta),1)
    
    X=np.insert(X , [0] , [1] ,1)
    
    h = X.dot(theta)
    J = 1/(2*m)* (np.square(h - y)).sum() + l/(2*m)*(np.square(theta[1:])).sum()
    
    l_grad = (l/m)*theta
    l_grad[0]=0
    l_grad= l_grad.reshape(len(l_grad),1)
    
    grad = 1/m * ((X.T).dot(h - y) )+ l_grad
    
    return J , grad
    
J , grad= linearRegCostFunction(theta, X, y, 1 )
print("Cost at theta = [1 ; 1] ",J , "and gradient \n", grad,"\n")

results = op.minimize(fun=linearRegCostFunction ,x0= theta, args=(X, y , l) ,method = 'TNC', jac= True)
print("Cost at optimized theta ",results.fun , "and gradient \n", results.x,"\n")

theta_opt = results.x
theta_opt = theta_opt.reshape(len(theta_opt),1)
abc = np.insert(X , [0] , [1] ,1).dot(theta_opt)

plt.figure(1)
plt.plot(X , abc, c= 'b')   #blue line
plt.scatter(X , y , marker="x", c='r')   #red cross
plt.xlabel("Change in water level(X)")
plt.ylabel("Water flowing out of the damn(y)")   
plt.title("Linear fit")
#plt.close()   

#learning curves  , l is zero for evaluating error
def learningCurves(X , y , Xval , yval , l):
    
    m = len(y)
    error_train = np.zeros([m,1])
    error_val = np.zeros([m,1])
    theta = np.ones([X.shape[1]+1,1])
    
    for i in range(1,m+1):
        a=op.minimize(fun=linearRegCostFunction ,x0= theta, args=(X[:i], y[:i] , l)\
                      ,method = 'TNC', jac= True)
        theta=a.x
        error_train[i-1],c = linearRegCostFunction(theta , X[:i] , y[:i] , 0)
        error_val[i-1],e =linearRegCostFunction(theta , Xval, yval ,0)
        
    return error_train,error_val   
    
error_train , error_val = learningCurves(X , y , Xval , yval, l)

plt.figure(2)
p1, p2 =plt.plot(range(m) , error_train ,error_val)
plt.legend((p1,p2),("Train","Cross validation"))
plt.xlabel("Number of training examples(m)")
plt.ylabel("Error")
plt.title("Learning curve")
#plt.close()          

#polynomial regression
p=8
def polyFeatures(X, p):
    X_poly = np.zeros([len(X),p])
    
    for i in range(p):
        X_poly[0:,i] = np.power( X[0:,0] , i+1 )        
    return X_poly
    
def featureNormalize(X):
    mu = np.zeros([ 1, len(X[0])])
    sigma = np.zeros([1, len(X[0])])
    
    mu = [ X[:,i].mean() for i in range(len(X[0])) ]
    sigma = [ X[:,i].std() for i in range(len(X[0])) ]
    
    X_norm = (X-mu) /sigma

    return X_norm , mu , sigma

#Map X onto Polynomial Features and Normalize    
X_poly = polyFeatures(X ,p)    #adding polynomial features
X_poly , mu , sigma =featureNormalize(X_poly)   #normalizing 

#Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest ,p)
X_poly_test =( X_poly_test-mu)/sigma

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val =( X_poly_val - mu )/sigma

#learning curve for polynomial regression
l = 0
theta = np.ones([X_poly.shape[1]+1,1])
b = op.minimize(fun=linearRegCostFunction ,x0= theta, args=(X_poly, y , l)\
                      ,method = 'TNC', jac= True)
theta = b.x
theta = np.reshape(theta  , [len(theta),1])

def plotFit(min_x , max_x , mu , sigma , theta , p):
    x = np.arange(min_x -15, max_x+15 , 0.05)
    x = x.reshape([len(x),1])
    
    x_poly = polyFeatures(x ,p)    
    x_poly , mu , sigma =featureNormalize(x_poly) 
    x_poly = np.insert(x_poly , [0] , [1] ,1 )
    
    plt.plot(x , x_poly.dot(theta),'--', c='r' )
      
plt.figure(3)
plt.scatter(X , y , marker ='x')
plotFit(min(X) , max(X)  , mu , sigma , theta , p)
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');
plt.title ('Polynomial Regression Fit ')
#plt.close()

plt.figure(4)
error_train , error_val = learningCurves(X_poly , y , X_poly_val , yval, l)
p1, p2=plt.plot(range(m) , error_train , error_val)
plt.title('Polynomial Regression Learning Curve');
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend((p1,p2),('Train', 'Cross Validation'))
#plt.close()

for i in range(m):
    #print(i+1, error_train[i] , error_val[i])
    S=0 
    
def validationCurve(X , y , Xval , yval):
    
    lamda_vec =np.array([ [0],[0.001],[0.003],[0.01],[0.03],[0.1],[0.3],[1],[3],[10] ])
    error_train = np.zeros([len(lamda_vec),1])
    error_val = np.zeros([len(lamda_vec),1])
    
    for i in range(len(lamda_vec)):
        l = lamda_vec[i]
        theta = np.ones([X.shape[1]+1,1])
        b = op.minimize(fun=linearRegCostFunction ,x0= theta, args=(X, y , l)\
                      ,method = 'TNC', jac= True)
        theta = b.x
        
        error_train[i], dontNeedThis = linearRegCostFunction(theta  , X ,y , 0)
        error_val[i] , dontNeedThis = linearRegCostFunction(theta , Xval , yval , 0)
                    
    return error_train , error_val , lamda_vec
        
error_train , error_val , lamda_vec= validationCurve(X_poly , y , X_poly_val , yval)

plt.figure(5)
p1, p2=plt.plot(range(len(lamda_vec)) , error_train , error_val)
plt.xlabel('lambda')
plt.ylabel('Error')
plt.legend((p1,p2),('Train', 'Cross Validation'))
plt.title("Validation Curve")
#plt.close()

for i in range(len(lamda_vec)):
    #print(lamda_vec[i]   , error_train[i] ,  error_val[i])
    S=0 
