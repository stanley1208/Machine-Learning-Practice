import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

m=100
X=6*np.random.rand(m,1)-3
y=2+X+0.5*X**2+np.random.randn(m,1)

X_train,X_val,y_train,y_val=train_test_split(X[:50],y[:50].ravel(),test_size=0.5,random_state=42)
poly_scalar=Pipeline([
    ("poly_features",PolynomialFeatures(degree=90,include_bias=False)),
    ("std_scalar",StandardScaler())
])

X_train_poly_scaled=poly_scalar.fit_transform(X_train)
X_val_poly_scaled=poly_scalar.transform(X_val)

sgd_reg=SGDRegressor(max_iter=1,tol=-np.infty,warm_start=True,penalty=None,learning_rate="constant",eta0=0.0005,random_state=42)

minimun_val_error=float("inf")
best_epoch=None
best_model=None

# for epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled,y_train)    # continues where it left off
#     y_val_predict=sgd_reg.predict(X_val_poly_scaled)
#     val_error=mean_squared_error(y_val,y_val_predict)
#     if val_error < minimun_val_error:
#         minimun_val_error=val_error
#         best_epoch=epoch
#         best_model=deepcopy(sgd_reg)


sgd_reg=SGDRegressor(max_iter=1,tol=-np.infty,warm_start=True,penalty=None,learning_rate="constant",eta0=0.0005,random_state=42)


n_epochs=500
train_errors,val_errors=[],[]
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled,y_train)    # continues where it left off
    y_train_predict=sgd_reg.predict(X_train_poly_scaled)
    y_val_predict=sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train,y_train_predict))
    val_errors.append(mean_squared_error(y_val,y_val_predict))

best_epoch=np.argmin(val_errors)
best_val_rmse=np.sqrt(val_errors[best_epoch])

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 2),
             ha="center",
             arrowprops=dict(facecolor='cyan', shrink=0.005),
             fontsize=16,
            )

best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()

print(best_epoch,best_model)




