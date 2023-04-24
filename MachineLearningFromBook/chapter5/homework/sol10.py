from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal,uniform
import numpy as np

housing=fetch_california_housing()
X=housing["data"]
y=housing["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# lin_svr=LinearSVR(random_state=42)
# print(lin_svr.fit(X_train_scaled,y_train))

# y_pred=lin_svr.predict(X_train_scaled)
# mse=mean_squared_error(y_train,y_pred)
# print(mse)

param_distributions={"gamma":reciprocal(0.001,0.1),"C":uniform(1,10)}
rnd_search_cv=RandomizedSearchCV(SVR(),param_distributions,n_iter=10,verbose=2,cv=3)
rnd_search_cv.fit(X_train_scaled,y_train)

print(rnd_search_cv.best_estimator_)

y_pred=rnd_search_cv.best_estimator_.predict(X_train_scaled)
mse=mean_squared_error(y_train,y_pred)
rmse=np.sqrt(mse)
print(rmse)

y_pred=rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)


