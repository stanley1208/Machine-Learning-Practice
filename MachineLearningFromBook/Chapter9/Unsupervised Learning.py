import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


plt.rc('font', size=14)
plt.rc('axes', labelsize=14,titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Clustering

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


print((y_pred==y).sum()/len(y_pred))

# K-Means
blob_centers=np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                         [-2.8,  2.8], [-2.8,  1.3]])
blob_std=np.array([0.4,0.3,0.1,0.1,0.1])
X,y=make_blobs(n_samples=1000,centers=blob_centers,cluster_std=blob_std,random_state=7)
k=5
kmeans=KMeans(n_clusters=k,n_init=10,random_state=42)
y_pred=kmeans.fit_predict(X)

def plot_clusters(X,y=None):
    plt.scatter(X[:,0],X[:,1],c=y,s=1)
    plt.xlabel("$x_1")
    plt.ylabel("$x_2",rotation=0)

plt.figure(figsize=(10,5))
plot_clusters(X)
plt.gca().set_axisbelow(True)
plt.grid()
plt.show()

print(y_pred is kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

X_new=np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(kmeans.predict(X_new))

# Decision Boundaries
def plot_data(X):
    plt.plot(X[:,0],X[:,1],'k.',markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8,4))
plot_decision_boundaries(kmeans,X)
plt.show()

print(kmeans.transform(X_new).round(2))


kmeans_iter1=KMeans(n_clusters=5,init='random',n_init=1,max_iter=1,random_state=5)
kmeans_iter2=KMeans(n_clusters=5,init='random',n_init=1,max_iter=2,random_state=5)
kmeans_iter3=KMeans(n_clusters=5,init='random',n_init=1,max_iter=3,random_state=5)

kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)


