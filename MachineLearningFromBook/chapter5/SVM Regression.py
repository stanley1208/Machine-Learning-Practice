import numpy as np
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt


np.random.seed(42)
m=50
X=2*np.random.rand(m,1)
y=(4+3*X+np.random.randn(m,1)).ravel()

svm_reg1=LinearSVR(epsilon=1.5,random_state=42)
svm_reg2=LinearSVR(epsilon=0.5,random_state=42)
svm_reg1.fit(X,y)
svm_reg2.fit(X,y)

def find_support_vectors(svm_reg,X,y):
    y_pred=svm_reg.predict(X)
    off_margin=(np.abs(y-y_pred)>=svm_reg.epsilon)
    return np.argwhere(off_margin)

svm_reg1.support_=find_support_vectors(svm_reg1,X,y)
svm_reg2.support_=find_support_vectors(svm_reg2,X,y)

eps_x1=1
eps_y_pred=svm_reg1.predict([[eps_x1]])

def plot_svm_regression(svm_reg,X,y,axes):
    x1s=np.linspace(axes[0],axes[1],100).reshape(100,1)
    y_pred=svm_reg.predict(x1s)
    plt.plot(x1s,y_pred,"k-",linewidth=2,label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon,"k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_],y[svm_reg.support_],s=180,facecolor="#FFAAAA")
    plt.plot(X,y,"bo")
    plt.xlabel(r"$x_1$",fontsize=18)
    plt.legend(loc="upper left",fontsize=18)
    plt.axis(axes)

fig,axes=plt.subplots(ncols=2,figsize=(10,8),sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1,X,y,[0,2,3,11])
plt.title(r"$\epsilon={}$".format(svm_reg1.epsilon),fontsize=18)
plt.ylabel(r"$y$",fontsize=18,rotation=0)
plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)

plt.show()



