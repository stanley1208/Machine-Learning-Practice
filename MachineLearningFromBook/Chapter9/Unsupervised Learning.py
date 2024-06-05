import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

plt.rc('font', size=14)
plt.rc('axes', labelsize=14,titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


data=load_iris()
X=data.data
y=data.target
print(data.target_names)


plt.figure(figsize=(9,5))

plt.subplot(121)
plt.plot(X[y==0,2],X[y==0,3],'yo',label='Iris setosa')
plt.plot(X[y==1,2],X[y==1,3],'bs',label='Iris versicolor')
plt.plot(X[y==2,2],X[y==2,3],'g^',label='Iris virginica')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.grid()
plt.legend()

plt.subplot(122)
plt.scatter(X[:,2],X[:,3],c='k',marker='.')
plt.xlabel('Petal length')
plt.tick_params(labelleft=False)
plt.gca().set_axisbelow(True)
plt.grid()

plt.show()
# print(X)
# print(X[y==0,2])
# print(X[y==0,3])
# print(X[y==0,2],X[y==0,3])
# print(X[y==1,2],X[y==1,3])
# print(X[y==2,2],X[y==2,3])

# print(X[:,2],X[:,3])