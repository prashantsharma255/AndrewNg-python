import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm

#example on dataset 1
data1 = loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\ex6data1.mat")
X1 = data1["X"]
y1 = data1["y"]

#training svm
#clf = svm.SVC(kernel='linear',C=1 ).fit(X1,y1.ravel())

def plotGraph(X,y):
    pos= np.where(y==1)
    neg = np.where(y==0)
    plt.scatter( X[pos,0],X[pos,1] , marker="+" , c='black' )
    plt.scatter( X[neg[0],0],X[neg[0],1], marker="o",c='yellow')
 
def visualizeBoundaryLinear(X,y):    
    #create a mesh to plot in
    xmin , xmax = X[:,0].min()-0.025 , X[:,0].max()+0.025
    ymin , ymax = X[:,1].min()-0.025 , X[:,1].max()+0.025
    h = 0.01 #(xmax/xmin)/100
    xx,yy = np.meshgrid( np.arange(xmin,xmax,h) , np.arange(ymin,ymax,h) )

    z = clf.predict(np.c_[xx.ravel() , yy.ravel()])  #predicting for z and contour plotting
    z = z.reshape(xx.shape)  
    plt.contour(xx ,yy ,z)

#implementing gaussian kernel
x1=np.array([[1],[2],[1]])
x2=np.array([[0],[4],[-1]])
def gaussianKernel(x1,x2):
    sim=0
    nr=0
    sigma=2
    
    nr =np.sum( np.square( x1 - x2))
    sim = np.exp( -( nr /(2*np.square(sigma))) )   
    return sim
    
#sim=gaussianKernel(x1,x2)
#print(sim)       

#dataset2
data2=loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\ex6data2.mat")
X2=data2['X']
y2=data2['y']
#plotGraph(X2,y2)
#clf = svm.SVC(C=50, kernel='rbf',gamma=15).fit(X2,y2.ravel())   #fitting dataset2
#visualizeBoundaryLinear(X2,y2)                                  #boundary for dataset2 

#dataset3
data3= loadmat(r"C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\ex6data3.mat")
X3=data3['X']
y3=data3['y']
#plotGraph(X3,y3)
clf=svm.SVC(kernel='rbf',C=10,gamma=30).fit(X3,y3.ravel())
#visualizeBoundaryLinear(X3,y3)

Xval=data3['Xval']
yval=data3['yval']

z=clf.predict( np.c_[Xval[:,0] ,Xval[:,1]] )
acc = (z==yval.ravel()).mean()*100
#print(acc)





