import numpy as np
from sklearn import datasets
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

# X1D=np.linspace(-4,4,9).reshape(-1,1)
# X2D=np.c_[X1D,X1D**2]
# y=np.array([0,0,1,1,1,1,1,0,0])

# plt.figure(figsize=(10,6))
#
# plt.subplot(121)
# plt.grid(True,which='both')
# plt.axhline(y=0,color='k')
# plt.plot(X1D[:,0][y==0],np.zeros(4),"bs")
# plt.plot(X1D[:,0][y==1],np.zeros(5),"g^")
# plt.gca().get_yaxis().set_ticks([])
# plt.xlabel(r"$x_1$",fontsize=20)
# plt.axis([-4.5,4.5,-0.2,0.2])
#
# plt.subplot(122)
# plt.grid(True,which='both')
# plt.axhline(y=0,color='k')
# plt.axvline(x=0,color='k')
# plt.plot(X2D[:,0][y==0],X2D[:,1][y==0],"bs")
# plt.plot(X2D[:,0][y==1],X2D[:,1][y==1],"g^")
# plt.xlabel(r"$x_1$",fontsize=20)
# plt.ylabel(r"$x_2$",fontsize=20,rotation=0)
# plt.gca().get_yaxis().set_ticks([0,4,8,12,16])
# plt.plot([-4.5,4.5],[6.5,6.5],"r--",linewidth=3)
# plt.axis([-4.5,4.5,-1,17])
#
# plt.show()


X,y=make_moons(n_samples=100,noise=0.15,random_state=42)

def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"bs")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"g^")
    plt.axis(axes)
    plt.grid(True,which="both")
    plt.xlabel(r"$x_1$",fontsize=20)
    plt.ylabel(r"$x_2$",fontsize=20,rotation=0)

plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plt.show()

polynomial_svm_clf=Pipeline([
    ("poly_features",PolynomialFeatures(degree=3)),
    ("scalar",StandardScaler()),
    ("svm_clf",LinearSVC(C=10,loss="hinge",random_state=42))
])

print(polynomial_svm_clf.fit(X,y))

def plot_predictions(clf,axes):
    x0s=np.linspace(axes[0],axes[1],100)
    x1s=np.linspace(axes[2],axes[3],100)
    x0,x1=np.meshgrid(x0s,x1s)
    X=np.c_[x0.ravel(),x1.ravel()]
    print(X)
    y_pred=clf.predict(X).reshape(x0.shape)
    print(y_pred)
    y_decision=clf.decision_function(X).reshape(x0.shape)
    print(y_decision)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)
    plt.contourf(x0,x1,y_decision,cmap=plt.cm.brg,alpha=0.1)

plot_predictions(polynomial_svm_clf,[-1.5,2.5,-1,1.5])
plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plt.show()






