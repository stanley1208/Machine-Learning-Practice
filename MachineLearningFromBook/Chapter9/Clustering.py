from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


data=load_iris()
X=data.data
y=data.target
print(data.target_names)


def plot_data(X):
    plt.plot(X[:,0],X[:,1],'k.',markersize=2)

def plot_centroids(centroids,weights=None,circle_color='w',cross_color='k'):
    if weights is not None:
        centroids=centroids[weights>weights.max()/10]
    plt.scatter(centroids[:,0],centroids[:,1],marker='o',s=35,linewidth=8,color=circle_color,zorder=10,alpha=0.9)
    plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=2,linewidth=12,color=cross_color,zorder=11,alpha=1)

def plot_decision_boundaries(clusterer,X,resolution=10000,show_centroids=True,show_xlabels=True,show_ylabels=True):
    mins=X.min(axis=0)-0.1
    maxs=X.max(axis=0)+0.1
    xx,yy=np.meshgrid(np.linspace(mins[0],maxs[0],resolution),
                      np.linspace(mins[1], maxs[1], resolution))
    Z=clusterer.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


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

# plt.plot(X[y_pred==0,2],X[y_pred==0,3],'yo',label='Cluster1')
# plt.plot(X[y_pred==1,2],X[y_pred==1,3],'bs',label='Cluster2')
# plt.plot(X[y_pred==2,2],X[y_pred==2,3],'g^',label='Cluster3')
# plt.xlabel("Patel length",fontsize=18)
# plt.ylabel("Patel width",fontsize=18)
# plt.legend(loc='best',fontsize=18)
# plt.show()

print(np.sum(y_pred==y))
print(np.sum(y_pred==y)/len(y_pred))

blob_centers=np.array(
    [[0.2,2.3],
     [-1.5,2.3],
     [-2.8,1.8],
     [-2.8,2.8],
     [-2.8,1.3]]
)
blob_std=np.array([0.4,0.3,0.1,0.1,0.1])
X,y=make_blobs(n_samples=2000,centers=blob_centers,cluster_std=blob_std,random_state=7)

def plot_cluster(X,y=None):
    plt.scatter(X[:,0],X[:,1],c=y,s=1)
    plt.xlabel("$x_1$",fontsize=14)
    plt.ylabel("$y_1$",fontsize=14,rotation=0)

plt.figure(figsize=(10,8))
plot_cluster(X)
plt.show()


k=5
kmeans=KMeans(n_clusters=k,random_state=42)
y_pred=kmeans.fit_predict(X)

print(y_pred)
print(y_pred is kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

X_new=np.array([[0,2],[3,2],[-3,3],[-3,2.5]])
print(kmeans.predict(X_new))


plt.figure(figsize=(10,8))
plot_decision_boundaries(kmeans,X)
plt.show()
