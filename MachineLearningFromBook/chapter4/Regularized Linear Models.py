import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Ridge Regression
m=20
X=3*np.random.rand(m,1)
y=1+0.5*X+np.random.randn(m,1)/1.5
X_new=np.linspace(0,3,100).reshape(100,1)

# ridge_reg=Ridge(alpha=1,solver="cholesky",random_state=42)
# ridge_reg.fit(X,y)
# print(ridge_reg.predict([[1.5]]))

# ridge_reg=Ridge(alpha=1,solver="sag",random_state=42)
# ridge_reg.fit(X,y)
# print(ridge_reg.predict([[1.5]]))

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model=model_class(alpha,**model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model=Pipeline([
                ("poly_features", PolynomialFeatures(degree=10,include_bias=False)),
                ("std_scalar", StandardScaler()),
                ("regul_reg", model),
            ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new,y_new_regul,style,linewidth=lw,label=r"$\alpha={}$".format(alpha))
    plt.plot(X,y,"b.",linewidth=3)
    plt.legend(loc="upper left",fontsize=18)
    plt.xlabel("$x_1$",fontsize=18)
    plt.axis([0,3,0,4])


# plt.figure(figsize=(8,4))
# plt.subplot(121)
# plot_model(Ridge, polynomial=False, alphas=(0,10,100), random_state=42)
# plt.ylabel("$y$",rotation=0,fontsize=18)
# plt.subplot(122)
# plot_model(Ridge, polynomial=True, alphas=(0,10**-5,1), random_state=42)
# plt.show()

# sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))

# Lasso Regression
plt.plot(figsize=(8,4))
plt.subplot(121)
plot_model(Lasso,polynomial=False,alphas=(0,0.1,1),random_state=42)
plt.ylabel("$y$",rotation=0,fontsize=18)
plt.subplot(122)
plot_model(Lasso,polynomial=True,alphas=(0,10**-7,1),random_state=42)
plt.show()

# lasso_reg=Lasso(alpha=0.1)
# lasso_reg.fit(X,y)
# print(lasso_reg.predict([[1.5]]))
#
# sgd_reg = SGDRegressor(penalty="l1", max_iter=1000, tol=1e-3, random_state=42)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))

# Elastic Net
# elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5)
# elastic_net.fit(X,y)
# print(elastic_net.predict([[1.5]]))


