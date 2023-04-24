from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(regressors,X,y,axes,label=None,style="r-",data_style="b.",data_label=None):
    x1=np.linspace(axes[0],axes[1],500)
    y_pred=sum(regressor.predict(x1.reshape(-1,1)) for regressor in regressors)
    plt.plot(X[:,0],y,data_style,label=data_label)
    plt.plot(x1,y_pred,style,linewidth=2,label=label)
    if label or data_label:
        plt.legend(loc="upper center",fontsize=16)
    plt.axis(axes)


np.random.seed(42)
X=np.random.rand(100,1)-0.5
y=3*X[:,0]**2+0.05*np.random.rand(100)


gbrt=GradientBoostingRegressor(max_depth=2,n_estimators=3,learning_rate=1.0,random_state=42)
gbrt.fit(X,y)

gbrt_slow=GradientBoostingRegressor(max_depth=2,n_estimators=200,learning_rate=0.1,random_state=42)
gbrt_slow.fit(X,y)

fix,axes=plt.subplots(ncols=2,figsize=(10,8),sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt],X,y,[-0.5,0.5,-0.1,0.8],label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate,gbrt.n_estimators),fontsize=14)
plt.xlabel("$x_1$",fontsize=16)
plt.ylabel("$y$",fontsize=16,rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_slow],X,y,[-0.5,0.5,-0.1,0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate,gbrt_slow.n_estimators),fontsize=14)
plt.xlabel("$x_1$",fontsize=16)

plt.show()


