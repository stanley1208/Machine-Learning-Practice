from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt



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
