import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


X,y=make_moons(n_samples=1000,noise=0.30,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42,n_jobs=-1)
rnd_clf.fit(X_train,y_train)

y_pred_rf=rnd_clf.predict(X_test)

bag_clf=BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt",max_leaf_nodes=16),n_estimators=500,random_state=42
)

bag_clf.fit(X_train,y_train)
y_pred_bag=bag_clf.predict(X_test)

print(np.sum(y_pred_bag==y_pred_rf)/len(y_pred_bag))    # very similar predictions

extra_clf=ExtraTreesClassifier(
    n_estimators=500,max_leaf_nodes=16,random_state=42,n_jobs=-1
)

extra_clf.fit(X_train,y_train)
y_pred_extra=extra_clf.predict(X_test)
print(accuracy_score(y_test,y_pred_extra))