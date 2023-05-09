from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt


X,t=make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
# rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
# X_reduced=rbf_pca.fit_transform(X)


lin_pca=KernelPCA(n_components=2,kernel='linear',fit_inverse_transform=True)
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04,fit_inverse_transform=True)
sig_pca=KernelPCA(n_components=2,kernel='sigmoid',gamma=0.0001,fit_inverse_transform=True)


plt.figure(figsize=(12,6))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced=pca.fit_transform(X)
    if subplot==132:
        X_reduced_rbf=X_reduced

    plt.subplot(subplot)
    plt.title(title,fontsize=18)
    plt.scatter(X_reduced[:,0],X_reduced[:,1],c=t,cmap=plt.cm.hot)
    plt.xlabel("$x_1$",fontsize=18)
    if subplot==131:
        plt.ylabel("$y_1$",fontsize=18,rotation=0)
    # plt.grid(True)

plt.show()