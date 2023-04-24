import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_digit(data):
    image=data.reshape(28,28)
    plt.imshow(image,cmap=mpl.cm.hot,
               interpolation="nearest")
    plt.axis("off")


mnist=fetch_openml('mnist_784',version=1,as_frame=False)
mnist.target=mnist.target.astype(np.uint8)

rnd_clf=RandomForestClassifier(n_estimators=500,random_state=42)
rnd_clf.fit(mnist["data"],mnist["target"])

plot_digit(rnd_clf.feature_importances_)

cbar=plt.colorbar(ticks=[rnd_clf.feature_importances_.min(),rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important','Very important'])
plt.show()


