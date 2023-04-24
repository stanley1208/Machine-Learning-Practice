from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
mnist.target=mnist.target.astype(np.uint8)

X_train_val,X_test,y_train_val,y_test=train_test_split(mnist.data,mnist.target,test_size=10000,random_state=42)
# print(len(X_train_val))
# print(len(X_test_val))
# print(len(y_train_val))
# print(len(y_test_val))
X_train,X_val,y_train,y_val=train_test_split(X_train_val,y_train_val,test_size=10000,random_state=42)
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

random_forest_clf=RandomForestClassifier(n_estimators=100,random_state=42)
extra_trees_clf=ExtraTreesClassifier(n_estimators=100,random_state=42)
svm_clf=LinearSVC(max_iter=100,tol=20,random_state=42)
mlp_clf=MLPClassifier(random_state=42)


estimators=[random_forest_clf,extra_trees_clf,svm_clf,mlp_clf]
# for estimator in estimators:
#     print("Training the",estimator)
#     estimator.fit(X_train,y_train)

# [print(estimator.score(X_val,y_val)) for estimator in estimators]

named_estimators=[
    ("random_forest_clf",random_forest_clf),
    ("extra_trees_clf",extra_trees_clf),
    ("svm_clf",svm_clf),
    ("mlp_clf",mlp_clf),
]

voting_clf=VotingClassifier(named_estimators)
print(voting_clf.fit(X_train,y_train))
print(voting_clf.score(X_val,y_val))

a=[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
print(a)

voting_clf.set_params(svm_clf=None)

print(voting_clf.estimators)

print(voting_clf.estimators_)

del voting_clf.estimators_[2]

print(voting_clf.score(X_val,y_val))

voting_clf.voting="soft"

print(voting_clf.score(X_val,y_val))

voting_clf.voting="hard"

print(voting_clf.score(X_test,y_test))

b=[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
print(b)









