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

# heads_proba=0.51
# coin_tosses=(np.random.rand(10000,10)<heads_proba).astype(np.int32)
# cumulative_heads_ratio=np.cumsum(coin_tosses,axis=0)/np.arange(1,10001).reshape(-1,1)

# plt.figure(figsize=(8,4))
# plt.plot(cumulative_heads_ratio)
# plt.plot([0,10000],[0.51,0.51],"k--",linewidth=2,label="51%")
# plt.plot([0,10000],[0.5,0.5],"k-",label="50%")
# plt.xlabel("Number of coin tosses")
# plt.ylabel("Heads ratio")
# plt.legend(loc="best")
# plt.axis([0,10000,0.42,0.58])
# plt.show()

X,y=make_moons(n_samples=1000,noise=0.30,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

log_clf=LogisticRegression(solver='lbfgs',random_state=42)
rnd_clf=RandomForestClassifier(n_estimators=100,random_state=42)
svm_clf=SVC(gamma='scale',random_state=42,probability=True)

voting_clf=VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    voting='soft'
)

voting_clf.fit(X_train,y_train)

for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))




