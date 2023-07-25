from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


data=load_iris()
X=data.data
y=data.target
print(data.target_names)


plt.figure(figsize=(10,8))

plt.subplot(121)
plt.plot(X[y==0,2],X[y==0,3],'yo',label='Iris setosa')
plt.plot(X[y==1,2],X[y==1,3],'bs',label='Iris versicolor')
plt.plot(X[y==2,2],X[y==2,3],'g^',label='Iris virginica')
plt.xlabel("Patel length",fontsize=18)
plt.ylabel("Patel width",fontsize=18)
plt.legend(fontsize=18)


plt.subplot(122)
plt.scatter(X[:,2],X[:,3],c="k",marker=".")
plt.xlabel("Patel length",fontsize=18)


plt.show()
