import numpy as np
from sklearn import datasets
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def plot_svc_decision_boundary(svm_clf,xmin,xmax):
    w=svm_clf.coef_[0]
    b=svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0=np.linspace(xmin,xmax,200)
    decision_boundary=-w[0]/w[1]*x0-b/w[1]

    margin=1/w[1]
    gutter_up=decision_boundary+margin
    gutter_down=decision_boundary-margin

    svs=svm_clf.support_vectors_
    plt.scatter(svs[:,0],svs[:,1],s=100,facecolors='#FFAAAA')
    plt.plot(x0,decision_boundary,"k-",linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


iris=datasets.load_iris()
X=iris["data"][:,(2,3)] # petal length, petal width
y=(iris["target"]==2).astype(np.float64) # Iris virginica

svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("linear_svc",LinearSVC(C=1,loss="hinge",random_state=42)),
])

# print(svm_clf.fit(X,y))
# print(svm_clf.predict([[5.5,2.7]]))

scaler=StandardScaler()
svm_clf1=LinearSVC(C=1,loss="hinge",random_state=42)
svm_clf2=LinearSVC(C=100,loss="hinge",random_state=42)

scaled_svm_clf1=Pipeline([
    ("scaler",scaler),
    ("linear_svc",svm_clf1)
])

scaled_svm_clf2=Pipeline([
    ("scaler",scaler),
    ("linear_svc",svm_clf2)
])

print(scaled_svm_clf1.fit(X,y))
print(scaled_svm_clf2.fit(X,y))


# Convert to unscaled parameters
b1=svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2=svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1=svm_clf1.coef_[0]/scaler.scale_
w2=svm_clf2.coef_[0]/scaler.scale_
svm_clf1.intercept_=np.array([b1])
svm_clf2.intercept_=np.array([b2])
svm_clf1.coef_=np.array([w1])
svm_clf2.coef_=np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
t=y*2-1
support_vectors_idx1=(t*(X.dot(w1)+b1)<1).ravel()
support_vectors_idx2=(t*(X.dot(w2)+b2)<1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]


fig,axes=plt.subplots(ncols=2,figsize=(10,5),sharey=True)

plt.sca(axes[0])
plt.plot(X[:,0][y==1],X[:,1][y==1],"g^",label="Iris virginica")
plt.plot(X[:,0][y==0],X[:,1][y==0],"bs",label="Iris versicolor")
plot_svc_decision_boundary(svm_clf1,4,5.9)
plt.xlabel("Patel length",fontsize=14)
plt.ylabel("Patel width",fontsize=14)
plt.legend(loc="best",fontsize=14)
plt.title("$C={}$".format(svm_clf1.C),fontsize=16)
plt.axis([4,5.9,0.8,2.8])

plt.sca(axes[1])
plt.plot(X[:,0][y==1],X[:,1][y==1],"g^",)
plt.plot(X[:,0][y==0],X[:,1][y==0],"bs",)
plot_svc_decision_boundary(svm_clf2,4,5.99)
plt.xlabel("Patel length",fontsize=14)
plt.legend(loc="best",fontsize=14)
plt.title("$C={}$".format(svm_clf2.C),fontsize=16)
plt.axis([4,5.9,0.8,2.8])

plt.show()





