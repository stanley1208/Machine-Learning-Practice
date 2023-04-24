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

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
# print(mnist.keys())

X,y=mnist["data"],mnist["target"]
# print(X.shape)
# print(y.shape)

some_digit = X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
# plt.show()

# print(y[0])

y=y.astype(np.uint8)

X_train,x_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5=(y_train==5) # 所有的5都是 True，所有其他數字都是False
y_test_5=(y_test==5)

# sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
# print(sgd_clf.fit(X_train,y_train_5))

# print(sgd_clf.predict([some_digit]))

skfolds=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

# for train_index,test_index in skfolds.split(X_train,y_train_5):
#     clone_clf=clone(sgd_clf)
#     X_train_folds=X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds,y_train_folds)
#     y_pred=clone_clf.predict(X_test_fold)
#     n_correct=sum(y_pred==y_test_fold)
#     print(n_correct/len(y_pred)) # 印出 0.9669, 0.91625與0.96785
#
# print(cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

# never_5_clf=Never5Classifier()
# print(cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

# y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
# print(confusion_matrix(y_train_5,y_train_pred))
# y_train_perfect_predictions=y_train_5 # 假裝我們得到完美的結果
# print(confusion_matrix(y_train_5,y_train_perfect_predictions))

# print(precision_score(y_train_5,y_train_pred))
# print(recall_score(y_train_5,y_train_pred))
# print(f1_score(y_train_5,y_train_pred))

# y_scores=sgd_clf.decision_function([some_digit])
# print(y_scores)
# threshold=0
# y_some_digit_pred=(y_scores>threshold)
# print(y_some_digit_pred)
# threshold=8000
# y_some_digit_pred=(y_scores>threshold)
# print(y_some_digit_pred)

# y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
# precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])



# recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
# threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


# plt.figure(figsize=(8, 4))
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
# plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
# plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
# plt.plot([threshold_90_precision], [0.9], "ro")
# plt.plot([threshold_90_precision], [recall_90_precision], "ro")
# plt.show()

# print((y_train_pred == (y_scores > 0)).all())

# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall", fontsize=16)
#     plt.ylabel("Precision", fontsize=16)
#     plt.axis([0, 1, 0, 1])
#     plt.grid(True)
#
# plt.figure(figsize=(8, 6))
# plot_precision_vs_recall(precisions, recalls)
# plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
# plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
# plt.plot([recall_90_precision], [0.9], "ro")
# plt.show()

# y_train_pred_90=(y_scores>=threshold_90_precision)

# print(precision_score(y_train_5,y_train_pred_90))
# print(recall_score(y_train_5,y_train_pred_90))

# fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
#
# plt.figure(figsize=(8, 6))
# plot_roc_curve(fpr, tpr)
# fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
# plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
# plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
# plt.plot([fpr_90], [recall_90_precision], "ro")
# plt.show()

forest_clf=RandomForestClassifier(random_state=42)
y_probas_forest=cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method="predict_proba")

y_scores_forest=y_probas_forest[:,1] # 分數=陽性類別的機率
fpr_forest,tpr_forest,thresholds_forest=roc_curve(y_train_5,y_scores_forest)


# recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]


# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
# plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
# plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
# plt.plot([fpr_90], [recall_90_precision], "ro")
# plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
# plt.plot([fpr_90], [recall_for_forest], "ro")
# plt.grid(True)
# plt.legend(loc="lower right", fontsize=16)
# plt.show()

# print(roc_auc_score(y_train_5,y_scores_forest))
# y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
# print(precision_score(y_train_5, y_train_pred_forest))
# print(recall_score(y_train_5, y_train_pred_forest))

# SVC()
# svm_clf=SVC()
# svm_clf.fit(X_train,y_train) # y_train not y_train_5
# # print(svm_clf.predict([some_digit])) # 輸出array([5],dtype=uint8)
#
# some_digit_scores=svm_clf.decision_function([some_digit])
# print(some_digit_scores)
# #[[ 1.72501977  2.72809088  7.2510018   8.3076379  -0.31087254  9.3132482
#    1.70975103  2.76765202  6.23049537  4.84771048]]
# np.argmax(some_digit_scores) # 5
# svm_clf.classes_ # array([0,1,2,3,4,5,6,7,8,9],dtype=uint8)
# svm_clf.classes_[5] # 5


# OneVsRestClassifier()
# ovr_clf=OneVsRestClassifier(SVC())
# ovr_clf.fit(X_train,y_train)
# print(ovr_clf.predict([some_digit])) # 輸出array([5],dtype=uint8)
# print(len(ovr_clf.estimators_)) # 10
#
# SGDClassifier()
# sgd_clf=SGDClassifier()
# sgd_clf.fit(X_train,y_train)
# print(sgd_clf.predict([some_digit])) # 輸出array([5],dtype=uint8)
# sgd_clf.decision_function([some_digit]) #
# array([[-31893.03095419, -34419.69069632,  -9530.63950739,
#           1823.73154031, -22320.14822878,  -1385.80478895,
#         -26188.91070951, -16147.51323997,  -4604.35491274,
#         -12050.767298  ]])
# cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.87365, 0.85835, 0.8689 ])
#
#
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# array([0.8983, 0.891 , 0.9018])
#
#
# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# conf_mx
# array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
#        [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
#        [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
#        [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
#        [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
#        [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
#        [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
#        [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
#        [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
#        [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]])
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()







