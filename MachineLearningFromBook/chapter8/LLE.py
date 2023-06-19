from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
mnist.target=mnist.target.astype(np.int8)



X,t=make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10)
X_reduced=lle.fit_transform(X)


plt.title("Unrolled swiss roll using LLE",fontsize=14)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=t,cmap=plt.cm.hot)
plt.xlabel("$z_1$",fontsize=18)
plt.ylabel("$z_2$",fontsize=18)
plt.axis([-0.065,0.055,-0.1,0.12])
plt.grid(True)
plt.show()


# Other Dimensionality Reduction Techniques
# Multidimensional Scaling (MDS)
mds=MDS(n_components=2,random_state=42)
X_reduced_mds=mds.fit_transform(X)

# Isomap
isomap=Isomap(n_components=2)
X_reduced_isomap=isomap.fit_transform(X)

# t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne=TSNE(n_components=2,random_state=42)
X_reduced_tsne=tsne.fit_transform(X)

# Linear Discriminant Analysis (LDA)
lda=LinearDiscriminantAnalysis(n_components=2)
X_mnist=mnist['data']
y_mnist=mnist['target']
lda.fit_transform(X_mnist,y_mnist)
X_reduced_lda=lda.transform(X_mnist)


titles=["MDS","Isomap","t-SNE"]
plt.figure(figsize=(10,8))

for subplot,title, X_reduced in zip((131,132,133),titles,
                                (X_reduced_mds,X_reduced_isomap,X_reduced_tsne)):

    plt.subplot(subplot)
    plt.title(title,fontsize=14)
    plt.scatter(X_reduced[:,0],X_reduced[:,1],c=t,cmap=plt.cm.hot)
    plt.xlabel("$z_1$",fontsize=18)
    if subplot==131:
        plt.ylabel("$z_2$",fontsize=18,rotation=0)

    plt.grid(True)
plt.show()












