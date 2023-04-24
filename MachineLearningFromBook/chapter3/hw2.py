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
from scipy.ndimage.interpolation import shift

def shift_image(image,dx,dy):
    image=image.reshape((28,28))
    shifted_image=shift(image,[dx,dy],cval=0,mode="constant")
    return shifted_image.reshape([-1])

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
X,y=mnist["data"],mnist["target"]

X_train,x_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

image = X_train[1000]
shifted_image_down = shift_image(image, 0, -5)
shifted_image_left = shift_image(image, -5, 0)

# plt.figure(figsize=(12,3))
# plt.subplot(131)
# plt.title("Original", fontsize=14)
# plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.subplot(132)
# plt.title("Shifted down", fontsize=14)
# plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.subplot(133)
# plt.title("Shifted left", fontsize=14)
# plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.show()

X_train_argumented=[image for image in X_train]
y_train_argumented=[label for label in y_train]

# Data Augmentation

for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
    for image,label in zip(X_train,y_train):
        X_train_argumented.append(shift_image(image,dx,dy))
        y_train_argumented.append(label)

X_train_argumented=np.array(X_train_argumented)
y_train_argumented=np.array(y_train_argumented)

shuffle_idx=np.random.permutation(len(X_train_argumented))
X_train_argumented=X_train_argumented[shuffle_idx]
y_train_argumented=y_train_argumented[shuffle_idx]

knn_clf=KNeighborsClassifier()
print(knn_clf.fit(X_train_argumented,y_train_argumented))

y_pred=knn_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

