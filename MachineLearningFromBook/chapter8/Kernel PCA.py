import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


X,t=make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
# rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
# X_reduced=rbf_pca.fit_transform(X)


lin_pca=KernelPCA(n_components=2,kernel='linear',fit_inverse_transform=True)
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04,fit_inverse_transform=True)
sig_pca=KernelPCA(n_components=2,kernel='sigmoid',gamma=0.00000001,fit_inverse_transform=True)

y=t>6.9

# plt.figure(figsize=(12,6))
# for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
#     X_reduced=pca.fit_transform(X)
#     if subplot==132:
#         X_reduced_rbf=X_reduced
#
#     plt.subplot(subplot)
#     plt.title(title,fontsize=18)
#     plt.scatter(X_reduced[:,0],X_reduced[:,1],c=t,cmap=plt.cm.hot)
#     plt.xlabel("$x_1$",fontsize=18)
#     if subplot==131:
#         plt.ylabel("$y_1$",fontsize=18,rotation=0)
#     # plt.grid(True)
#
# plt.show()


# plt.figure(figsize=(10,8))
#
# X_inverse=rbf_pca.inverse_transform(X_reduced_rbf)
#
# ax=plt.subplot(111,projection='3d')
# # ax.view_init(10,-70)
# ax.scatter(X_inverse[:,0],X_inverse[:,1],X_inverse[:,2],c=t,cmap=plt.cm.hot,marker='+')
# ax.set_xlabel("")
# ax.set_ylabel("")
# ax.set_zlabel("")
# # ax.set_xticklabels([])
# # ax.set_yticklabels([])
# # ax.set_zticklabels([])
#
# plt.show()


# X_inverse=rbf_pca.fit_transform(X)
#
# plt.figure(figsize=(10,8))
# plt.subplot(132)
# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=t,cmap=plt.cm.hot,marker='*')
# plt.xlabel("$z_1$",fontsize=18)
# plt.ylabel("$z_2$",fontsize=18,rotation=0)
# plt.show()


clf=Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(solver='lbfgs'))
    ])

param_grid=[{
    "kpca__gamma": np.linspace(0.03,0.05,10),
    "kpca__kernel": ['rbf','sigmoid']
    }]

grid_search=GridSearchCV(clf,param_grid,cv=3)
print(grid_search.fit(X,y))

print(grid_search.best_params_)

rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.0433,fit_inverse_transform=True)

X_reduced=rbf_pca.fit_transform(X) # kernel trick

X_preimage=rbf_pca.inverse_transform(X_reduced)

print(mean_squared_error(X,X_preimage))



