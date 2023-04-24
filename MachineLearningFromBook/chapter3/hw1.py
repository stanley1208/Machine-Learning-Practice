from math import ceil
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# An MNIST Classifier With Over 97% Accuracy

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
X,y=mnist["data"],mnist["target"]

some_digit = X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")

y=y.astype(np.uint8)

X_train,x_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

param_grid=[{"weights":["uniform","distance"],'n_neighbors':[3,4,5]}]
knn_clf=KNeighborsClassifier()
grid_search=GridSearchCV(knn_clf,param_grid,cv=5,verbose=3)
print(grid_search.fit(X_train,y_train))
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred=grid_search.predict(x_test)
print(accuracy_score(y_test,y_pred))
