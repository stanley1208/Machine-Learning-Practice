import os
import tarfile
import urllib.request

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)

import numpy as np

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon,reciprocal
DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH=os.path.join("datasets", "housing")
HOUSING_URL=DOWNLOAD_ROOT+"datasets/housing/housing.tgz"

# Where to save the figures
PROJECT_ROOT_DIR = ""
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

room_ix,bedroom_ix,population_ix,household_ix=3,4,5,6

class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # 沒有*args 或 **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # 沒有其他工作了

    def transform(self, X):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:, bedroom_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32

def split_train_test_by_id(data,test_ratio,id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.iloc[~in_test_set], data.iloc[in_test_set]

fetch_housing_data()
housing=load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=500,figsize=(20,15))
# plt.show()

# train_set,test_set=split_train_test(housing,0.2)
# print(len(train_set))
# print(len(test_set))

# train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
housing['income_cat']=pd.cut(housing['median_income'],
                             bins=[0,1.5,3.0,4.5,6,np.inf],
                             labels=[1,2,3,4,5])
# housing['income_cat'].hist()
# plt.show()

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

for set_ in  (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

housing=strat_train_set.copy()

# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
# plt.show()

# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
#              s=housing["population"]/100,label="population",figsize=(10,7),
#              c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
# plt.legend()
# plt.show()
# corr_matrix=housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
# scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()

# housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
# plt.show()

# housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix=housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

housing.dropna(subset=["total_bedrooms"]) # 捨棄相應的地區
housing.drop("total_bedrooms",axis=1) # 捨棄整個屬性
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True) # 設值為中位數

imputer=SimpleImputer(strategy="median")
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X=imputer.transform(housing_num)
# print(X)

housing_tr=pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
# print(housing_tr)

housing_cat=housing[["ocean_proximity"]]
# print(housing_cat.head(10))

ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])

# print(ordinal_encoder.categories_)
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)

# print(housing_cat_1hot.toarray())

# print(cat_encoder.categories_)

# attr_adder=CombineAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs=attr_adder.transform(housing.values)
# print(housing_extra_attribs)

num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attrib_adder',CombineAttributesAdder()),
    ('std_scalar',StandardScaler()),
])

housing_num_tr=num_pipeline.fit_transform(housing_num)

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

full_pipeline=ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',OneHotEncoder(),cat_attribs),
])

housing_prepared=full_pipeline.fit_transform(housing)
# print(housing_prepared)

lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
# print("Predictions:",lin_reg.predict(some_data_prepared))
# print("Labels:",list(some_labels))

housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
# print(lin_rmse)

tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels,housing_predictions)
tree_rmse=np.sqrt(tree_mse)
# print(tree_rmse)

scores=cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores=np.sqrt(-scores)
# display_scores(tree_rmse_scores)

lin_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores=np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)

# forest_reg=RandomForestRegressor()
# forest_reg.fit(housing_prepared,housing_labels)

# housing_predictions=forest_reg.predict(housing_prepared)
# forest_mse=mean_squared_error(housing_labels,housing_predictions)
# forest_rmse=np.sqrt(forest_mse)

# forest_scores=cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
# forest_rmse_scores=np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)



# joblib.dump(my_model,"my_model.pkl")
# my_model_loaded=joblib.load("my_model.pkl")

param_grid=[
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
]

forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)

grid_search.fit(housing_prepared,housing_labels)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

cvres=grid_search.cv_results_
# for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
#     print(np.sqrt(-mean_score),params)

feature_importances=grid_search.best_estimator_.feature_importances_
# print(feature_importances)

extra_attribs=["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
cat_encoder=full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs=list(cat_encoder.categories[0])
attributes=num_attribs+extra_attribs+cat_one_hot_attribs
# print(sorted(zip(feature_importances,attributes),reverse=True))

final_model=grid_search.best_estimator_

X_test=strat_test_set.drop("median_house_value",axis=1)
y_test=strat_test_set["median_house_value"].copy()

X_test_prepared=full_pipeline.transform(X_test)

final_predictions=final_model.predict(X_test_prepared)

final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)

confidence=0.95
squared_errors=(final_predictions-y_test)**2
# print(np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,loc=squared_errors.mean(),scale=stats.sem(squared_errors))))


#Exercise
# 1.
# param_grid2 = [
#         {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
#         {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
#          'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
#     ]
#
# svm_reg=SVR()
# grid_serach2=GridSearchCV(svm_reg,param_grid2,cv=5,scoring='neg_mean_squared_error',verbose=2)
# grid_serach2.fit(housing_prepared,housing_labels)
#
# negative_mse=grid_serach2.best_score_
# rmse=np.sqrt(-negative_mse)
# print(rmse)
# print(grid_serach2.best_params_)

# 2.
param_distribs={
    'kernel':['linear','rbf'],
    'C':reciprocal(20,200000),
    'gamma':expon(scale=1.0),
}

svm_reg2=SVR()
rnd_search=RandomizedSearchCV(svm_reg2,param_distributions=param_distribs,
                              n_iter=50,cv=5,scoring='neg_mean_squared_error',
                              verbose=2,random_state=42)
# print(rnd_search.fit(housing_prepared,housing_labels))
# negative_mse2=rnd_search.best_score_
# rmse2=np.sqrt(-negative_mse2)
# print(rmse2)
# print(grid_search.best_params_)

# expon_distrib = expon(scale=1.0)
# samples = expon_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Exponential distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()

# reciprocal_distrib = reciprocal(20, 200000)
# samples = reciprocal_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Reciprocal distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()

# 3.
def indices_of_top_k(arr,k):
    return np.sort(np.argpartition(np.array(arr),-k)[-k:])

class TopFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self,feature_importances,k):
        self.feature_importances=feature_importances
        self.k=k
    def fit(self,X,y=None):
        self.feature_indices_=indices_of_top_k(self.feature_importances,self.k)
        return self
    def transform(self):
        return X[:,self.feature_indices_]

k=5

top_k_feature_indices=indices_of_top_k(feature_importances,k)
# print(top_k_feature_indices)
# print(np.array(attributes)[top_k_feature_indices])
# print(sorted(zip(feature_importances,attributes),reverse=True)[:k])
# preparation_and_feature_selection_pipeline=Pipeline([
#     ('preparation',full_pipeline),
#     ('feature_selection',TopFeatureSelector(feature_importances,k))
# ])
# housing_prpared_top_k_features=preparation_and_feature_selection_pipeline.fit(housing)
# print(housing_prpared_top_k_features[0:3])
# print(housing_prepared[0:3,top_k_feature_indices])

# 4
prepare_select_and_predict_pipeline=Pipeline([
    ('preparation',full_pipeline),
    ('feature_selection',TopFeatureSelector(feature_importances,k)),
    ('svm_reg',SVR(**rnd_search.best_params_))
])
prepare_select_and_predict_pipeline.fit(housing,housing_labels)

some_data2=housing.iloc[:4]
some_labels2=housing_labels.iloc[:4]

print("Predictions:\t",prepare_select_and_predict_pipeline.predict(some_data2))
print("Labels:\t\t",list(some_labels2))

# 5
full_pipeline.named_transformers_["cat"].handle_unknown = 'ignore'

param_grid2 = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid2, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)

grid_search_prep.best_params_