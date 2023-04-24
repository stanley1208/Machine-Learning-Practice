import numpy as np
from sklearn import datasets
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures


def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"bs")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"g^")
    plt.axis(axes)
    plt.grid(True,which="both")
    plt.xlabel(r"$x_1$",fontsize=20)
    plt.ylabel(r"$x_2$",fontsize=20,rotation=0)

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

X,y=make_moons(n_samples=100,noise=0.15,random_state=42)

poly_kernel_svm_clf=Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel="poly",degree=3,coef0=1,C=5))
])

poly_kernel_svm_clf.fit(X,y)

poly100_kernel_svm_clf=Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel="poly",degree=10,coef0=100,C=5))
])

poly100_kernel_svm_clf.fit(X,y)

fig,axes=plt.subplots(ncols=2,figsize=(10,5),sharey=True)

plt.sca(axes[0])
plot_predictions(poly_kernel_svm_clf,[-1.5,2.45,-1,1.5])
plot_dataset(X,y,[-1.5,2.45,-1,1.5])
plt.title(r"$d=3, r=1, C=5$",fontsize=18)

plt.sca(axes[1])
plot_predictions(poly100_kernel_svm_clf,[-1.5,2.45,-1,1.5])
plot_dataset(X,y,[-1.5,2.45,-1,1.5])
plt.title(r"$d=10, r=100, C=5$",fontsize=18)
plt.ylabel("")

plt.show()