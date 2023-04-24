from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal,uniform

mnist=fetch_openml('mnist_784',version=1,cache=True,as_frame=False)
X=mnist["data"]
y=mnist["target"].astype(np.uint8)

X_train=X[:60000]
y_train=y[:60000]
X_test=X[60000:]
y_test=y[60000:]

# lin_clf=LinearSVC(random_state=42)
# lin_clf.fit(X_train,y_train)

# y_pred=lin_clf.predict(X_train)
# print(accuracy_score(y_train,y_pred))

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled=scaler.transform(X_test.astype(np.float32))

# lin_clf=LinearSVC(random_state=42)
# lin_clf.fit(X_train_scaled,y_train)
#
# y_pred=lin_clf.predict(X_train_scaled)
# print(accuracy_score(y_train,y_pred))
#
svm_clf=SVC(gamma="scale")
svm_clf.fit(X_train_scaled[:10000],y_train[:10000])
#
# y_pred=svm_clf.predict(X_train_scaled)
# print(accuracy_score(y_train,y_pred))

param_distributions={"gamma":reciprocal(0.001,0.1),"C":uniform(1,10)}
rnd_search_cv=RandomizedSearchCV(svm_clf,param_distributions,n_iter=10,verbose=2,cv=3)
print(rnd_search_cv.fit(X_train_scaled[:1000],y_train[:1000]))

print(rnd_search_cv.best_estimator_)
print(rnd_search_cv.best_score_)

print(rnd_search_cv.best_estimator_.fit(X_train_scaled,y_train))

y_pred=rnd_search_cv.best_estimator_.predict(X_train_scaled)
print(accuracy_score(y_train,y_pred))






