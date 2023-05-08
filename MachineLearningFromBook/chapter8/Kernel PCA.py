from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll


X,t=make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
X_rediced=rbf_pca.fit_transform(X)


