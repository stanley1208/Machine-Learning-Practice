import os
import urllib.request
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pylab as plt






TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"

def fetch_titantic_data(url=DOWNLOAD_URL,path=TITANIC_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
    for filename in ("train.csv","test.csv"):
        filepath=os.path.join(path,filename)
        if not os.path.isfile(filepath):
            print("Downloading",filename)
            urllib.request.urlretrieve(url+filename,filepath)

fetch_titantic_data()

def load_titantic_data(filename,titantic_path=TITANIC_PATH):
    csv_path=os.path.join(titantic_path,filename)
    return pd.read_csv(csv_path)

train_data=load_titantic_data("train.csv")
test_data=load_titantic_data("test.csv")

# print(train_data.head())

train_data=train_data.set_index("PassengerId")
test_data=test_data.set_index("PassengerId")

# print(train_data.info())

# print(train_data[train_data["Sex"]=="female"]["Age"].median())

# print(train_data.describe())

# print(train_data["Survived"].value_counts())

# print(train_data["Pclass"].value_counts())

# print(train_data["Sex"].value_counts())

# print(train_data["Embarked"].value_counts())

num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scalar",StandardScaler())
])

cat_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("cat_encoder",OneHotEncoder(sparse=False))
])

num_attribs=["Age","SibSp","Parch","Fare"]
cat_attribs=["Pclass","Sex","Embarked"]

preprocess_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs),
])

X_train=preprocess_pipeline.fit_transform(
    train_data[num_attribs+cat_attribs]
)

# print(X_train)

y_train=train_data["Survived"]

forest_clf=RandomForestClassifier(n_estimators=100,random_state=42)
print(forest_clf.fit(X_train,y_train))

X_test=preprocess_pipeline.transform(test_data[num_attribs+cat_attribs])
y_pred=forest_clf.predict(X_test)

forest_scores=cross_val_score(forest_clf,X_train,y_train,cv=10)
print(forest_scores.mean())

svm_clf=SVC(gamma="auto")
svm_scores=cross_val_score(svm_clf,X_train,y_train,cv=10)
print(svm_scores.mean())

# plt.figure(figsize=(8,4))
# plt.plot([1]*10,svm_scores,".")
# plt.plot([2]*10,forest_scores,".")
# plt.boxplot([svm_scores,forest_scores],labels=("SVM","Random Forest"))
# plt.ylabel("Accuracy",fontsize=14)
# plt.show()

train_data["AgeBucket"]=train_data["Age"]//15*15
print(train_data[["AgeBucket","Survived"]].groupby(["AgeBucket"]).mean())

train_data["RelativeOnboard"]=train_data["SibSp"]+train_data["Parch"]
print(train_data[["RelativeOnboard","Survived"]].groupby(["RelativeOnboard"]).mean())

