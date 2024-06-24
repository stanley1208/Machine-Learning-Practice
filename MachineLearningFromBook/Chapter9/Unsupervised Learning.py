import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from scipy import stats
from sklearn.mixture import GaussianMixture



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


y_pred=GaussianMixture(n_components=3,random_state=42).fit(X).predict(X)

mapping={}
for class_id in np.unique(y):
    mode,_=stats.mode(y_pred[y==class_id])
    mapping[mode]=class_id

y_pred=np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred==0,2],X[y_pred==0,3],'yo',label='Cluster 1')
plt.plot(X[y_pred==1,2],X[y_pred==1,3],'bs',label='Cluster 2')
plt.plot(X[y_pred==2,2],X[y_pred==2,3],'g^',label='Cluster 3')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='best')
plt.grid()
plt.show()