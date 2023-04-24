import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# t = np.linspace(-10, 10, 100)
# sig = 1 / (1 + np.exp(-t))
# plt.figure(figsize=(9, 6))
# plt.plot([-10, 10], [0, 0], "k-")
# plt.plot([-10, 10], [0.5, 0.5], "k:")
# plt.plot([-10, 10], [1, 1], "k:")
# plt.plot([0, 0], [-1.1, 1.1], "k-")
# plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
# plt.xlabel("t")
# plt.legend(loc="upper left", fontsize=20)
# plt.axis([-10, 10, -0.1, 1.1])
# plt.show()

iris=datasets.load_iris()
# print(list(iris.keys()))
# print(iris.DESCR)
X=iris["data"][:,3:]    # petal width
y=(iris["target"]==2).astype(np.int)    # 1 if Iris virginica, else 0

log_reg=LogisticRegression(solver="lbfgs",random_state=42)
log_reg.fit(X,y.ravel())

X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)

# plt.plot(X_new,y_proba[:,1],"g-",linewidth=2,label="Iris virginica")
# plt.plot(X_new,y_proba[:,0],"b--",linewidth=2,label="Not Iris virginica")
# plt.show()

decision_boundary=X_new[y_proba[:,1]>=0.5][0]

# plt.figure(figsize=(8,5))
# plt.plot(X[y==0],y[y==0],"bs")
# plt.plot(X[y==1],y[y==1],"g^")
# plt.plot([decision_boundary,decision_boundary],[-1,2],"k:",linewidth=2)
# plt.plot(X_new,y_proba[:,1],"g-",linewidth=2,label="Iris virginica")
# plt.plot(X_new,y_proba[:,0],"b--",linewidth=2,label="Not Iris virginica")
# plt.text(decision_boundary+0.02,0.15,"Decision boundary",fontsize=14,color="k",ha="center")
# plt.arrow(decision_boundary,0.08,-0.3,0,head_width=0.05,head_length=0.1,fc='b',ec='b')
# plt.arrow(decision_boundary,0.92,0.3,0,head_width=0.05,head_length=0.1,fc='g',ec='g')
# plt.xlabel("Petal width (cm)",fontsize=14)
# plt.ylabel("Probability",fontsize=14)
# plt.legend(loc="best",fontsize=14)
# plt.axis([0,3,-0.02,1.02])
# plt.show()

# print(decision_boundary)
# print(log_reg.predict([[1.7],[1.5]]))


# Softmax Regression
# X=iris["data"][:,(2,3)] # petal length, petal width
# y=(iris["target"]==2).astype(np.int)  # 1 if Iris virginica, else 0
#
# log_reg=LogisticRegression(solver="lbfgs",C=10**10,random_state=42)
# log_reg.fit(X,y)
#
# x0,x1=np.meshgrid(
#     np.linspace(2.9,7,500).reshape(-1,1),
#     np.linspace(0.8,2.7,200).reshape(-1,1),
# )
#
# X_new=np.c_[x0.ravel(),x1.ravel()]
# y_proba=log_reg.predict_proba(X_new)
#
# plt.figure(figsize=(10,6))
# plt.plot(X[y==0,0],X[y==0,1],"bs")
# plt.plot(X[y==1,0],X[y==1,1],"g^")
#
# zz=y_proba[:,1].reshape(x0.shape)
# contour=plt.contour(x0,x1,zz,cmap=plt.cm.brg)
#
# left_right=np.array([2.9,7])
# boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
#
# plt.clabel(contour,inline=1,fontsize=12)
# plt.plot(left_right,boundary,"k--",linewidth=3)
# plt.text(3.5,1.5,"Not Iris virginica",fontsize=14,color="b",ha="center")
# plt.text(6.5,2.3,"Iris virginica",fontsize=14,color="g",ha="center")
# plt.xlabel("Patel length",fontsize=14)
# plt.ylabel("Patel width",fontsize=14)
# plt.axis([2.9,7,0.8,2.7])
# plt.show()

X=iris["data"][:,(2,3)] # petal length, petal width
y=iris["target"]

softmax_reg=LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10,random_state=42)
softmax_reg.fit(X,y)

x0,x1=np.meshgrid(
    np.linspace(0,8,500).reshape(-1,1),
    np.linspace(0,3.5,200).reshape(-1,1),
)

X_new=np.c_[x0.ravel(),x1.ravel()]

y_proba=softmax_reg.predict_proba(X_new)
y_predict=softmax_reg.predict(X_new)

zz1=y_proba[:,1].reshape(x0.shape)
zz=y_predict.reshape(x0.shape)

plt.figure(figsize=(10,6))
plt.plot(X[y==2,0],X[y==2,1],"g^",label="Iris virginica")
plt.plot(X[y==1,0],X[y==1,1],"bs",label="Iris versicolor")
plt.plot(X[y==0,0],X[y==0,1],"yo",label="Iris setosa")

custom_cmap=ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0,x1,zz,cmap=custom_cmap)
contour=plt.contour(x0,x1,zz1,cmap=plt.cm.brg)
plt.clabel(contour,inline=1,fontsize=12)
plt.xlabel("Patel length",fontsize=14)
plt.ylabel("Patel width",fontsize=14)
plt.legend(loc="best",fontsize=14)
plt.axis([0,8,0,3.5])
plt.show()

print(softmax_reg.predict([[4,2]]))
print(softmax_reg.predict_proba([[1,2]]))








