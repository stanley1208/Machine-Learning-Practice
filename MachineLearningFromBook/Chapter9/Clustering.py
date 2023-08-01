from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np

data=load_iris()
X=data.data
y=data.target
print(data.target_names)


# plt.figure(figsize=(10,8))
#
# plt.subplot(121)
# plt.plot(X[y==0,2],X[y==0,3],'yo',label='Iris setosa')
# plt.plot(X[y==1,2],X[y==1,3],'bs',label='Iris versicolor')
# plt.plot(X[y==2,2],X[y==2,3],'g^',label='Iris virginica')
# plt.xlabel("Patel length",fontsize=18)
# plt.ylabel("Patel width",fontsize=18)
# plt.legend(fontsize=18)
#
#
# plt.subplot(122)
# plt.scatter(X[:,2],X[:,3],c="k",marker=".")
# plt.xlabel("Patel length",fontsize=18)

# plt.show()


y_pred=GaussianMixture(n_components=3,random_state=42).fit(X).predict(X)

mapping={}
for class_id in np.unique(y):
    mode,_=stats.mode(y_pred[y==class_id])
    mapping[mode[0]]=class_id


print(mode[0])
print(mapping[mode[0]])
print(mapping)

y_pred=np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred==0,2],X[y_pred==0,3],'yo',label='Cluster1')
plt.plot(X[y_pred==1,2],X[y_pred==1,3],'bs',label='Cluster2')
plt.plot(X[y_pred==2,2],X[y_pred==2,3],'g^',label='Cluster3')
plt.xlabel("Patel length",fontsize=18)
plt.ylabel("Patel width",fontsize=18)
plt.legend(loc='best',fontsize=18)
plt.show()

print(np.sum(y_pred==y))
print(np.sum(y_pred==y)/len(y_pred))
