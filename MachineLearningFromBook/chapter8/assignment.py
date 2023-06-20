from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



# question9
mnist=fetch_openml('mnist_784',version=1,as_frame=False)
mnist.target=mnist.target.astype(np.float)

X_train=mnist["data"][:60000]
y_train=mnist["target"][:60000]
X_test=mnist["data"][60000:]
y_test=mnist["target"][60000:]

rnd_clf=RandomForestClassifier(n_estimators=100,random_state=42)
t0=time.time()
rnd_clf.fit(X_train,y_train)
t1=time.time()

print("Training time: {:.2f}".format(t1-t0))

y_pred=rnd_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))


pca=PCA(n_components=0.95)
X_train_reduced=pca.fit_transform(X_train)


rnd_clf2=RandomForestClassifier(n_estimators=100,random_state=42)
t0=time.time()
rnd_clf2.fit(X_train_reduced,y_train)
t1=time.time()


print("Training time: {:.2f}".format(t1-t0))


X_test_reduced=pca.transform(X_test)
y_pred=rnd_clf2.predict(X_test_reduced)
print(accuracy_score(y_test,y_pred))


log_clf=LogisticRegression(multi_class="multinomial",solver='lbfgs',random_state=42)
t0=time.time()
log_clf.fit(X_train,y_train)
t1=time.time()

print("Training time: {:.2f}".format(t1-t0))

y_pred=log_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))


log_clf2=LogisticRegression(multi_class="multinomial",solver='lbfgs',random_state=42)
t0=time.time()
log_clf2.fit(X_train_reduced,y_train)
t1=time.time()

print("Training time: {:.2f}".format(t1-t0))

y_pred=log_clf2.predict(X_test_reduced)
print(accuracy_score(y_test,y_pred))


# question10
np.random.seed(42)
m=10000
idx=np.random.permutation(60000)[:m]

X=mnist['data'][idx]
y=mnist['target'][idx]

tsne=TSNE(n_components=2,random_state=42)
X_reduced=tsne.fit_transform(X)


plt.figure(figsize=(10,10))
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y,cmap='jet')
plt.axis('off')
plt.colorbar()
plt.show()






