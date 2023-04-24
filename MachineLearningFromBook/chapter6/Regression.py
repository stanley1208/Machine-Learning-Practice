import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = ""
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


iris=load_iris()
X=iris.data[:,2:] # petal length and width
y=iris.target
# print(X)
tree_clf=DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,y)

export_graphviz(
    tree_clf,
    out_file=os.path.join(IMAGES_PATH,"iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH,"iris_tree.dot"))

np.random.seed(42)
m=200
X=np.random.rand(m,1)
y=4*(X-0.5)**2
y=y+np.random.randn(m,1)/10

# tree_reg=DecisionTreeRegressor(max_depth=2,random_state=42)
# tree_reg.fit(X,y)


tree_reg1=DecisionTreeRegressor(random_state=42,max_depth=2)
tree_reg2=DecisionTreeRegressor(random_state=42,max_depth=3)
tree_reg1.fit(X,y)
tree_reg2.fit(X,y)

def plot_regression_predictions(tree_reg,X,y,axes=[0,1,-0.2,1],ylabel="$y$"):
    x1=np.linspace(axes[0],axes[1],500).reshape(-1,1)
    y_pred=tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$",fontsize=18)
    if ylabel:
        plt.ylabel(ylabel,fontsize=18,rotation=0)
    plt.plot(X,y,"b.")
    plt.plot(x1,y_pred,"r.-",linewidth=2,label=r"$\hat{y}$")


fig,axes=plt.subplots(ncols=2,figsize=(10,8),sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg1,X,y)
for split,style in ((0.1973,"k-"),(0.0917,"k--"),(0.7718,"k--")):
    plt.plot([split,split],[-0.2,1],style,linewidth=2)
plt.text(0.21,0.65,"Depth=0",fontsize=15)
plt.text(0.01,0.2,"Depth=1",fontsize=13)
plt.text(0.65,0.8,"Depth=1",fontsize=13)
plt.legend(loc="best",fontsize=18)
plt.title("max_depth=2",fontsize=14)

plt.sca(axes[1])
plot_regression_predictions(tree_reg2,X,y,ylabel=None)
for split,style in ((0.1973,"k-"),(0.0917,"k--"),(0.7718,"k--")):
    plt.plot([split,split],[-0.2,1],style,linewidth=2)
for split in (0.0458,0.1298,0.2873,0.9040):
    plt.plot([split,split],[-0.2,1],"k:",linewidth=2)
plt.text(0.3,0.5,"Depth=2",fontsize=13)
plt.title("max_depth=3",fontsize=14)
plt.show()

export_graphviz(
    tree_reg1,
    out_file=os.path.join(IMAGES_PATH,"regression_tree.dot"),
    feature_names=["x1"],
    rounded=True,
    filled=True
)
Source.from_file(os.path.join(IMAGES_PATH,"regression_tree.dot"))

tree_reg1=DecisionTreeRegressor(random_state=42)
tree_reg2=DecisionTreeRegressor(random_state=42,min_samples_leaf=10)
tree_reg1.fit(X,y)
tree_reg2.fit(X,y)

x1=np.linspace(0,1,500).reshape(-1,1)
y_pred1=tree_reg1.predict(x1)
y_pred2=tree_reg2.predict(x1)

fig,axes=plt.subplots(ncols=2,figsize=(10,8),sharey=True)
plt.sca(axes[0])
plt.plot(X,y,"b.")
plt.plot(x1,y_pred1,"r.-",linewidth=2,label=r"$\hat{y}$")
plt.axis([0,1,-0.2,1.1])
plt.xlabel("$x_1$",fontsize=18)
plt.ylabel("$y$",fontsize=18,rotation=0)
plt.legend(loc="upper center",fontsize=15)
plt.title("No restrictions",fontsize=14)

plt.sca(axes[1])
plt.plot(X,y,"b.")
plt.plot(x1,y_pred2,"r.-",linewidth=2,label=r"$\hat{y}$")
plt.axis([0,1,-0.2,1.1])
plt.xlabel("$x_1$",fontsize=18)
plt.title("min_sample_leaf={}".format(tree_reg2.min_samples_leaf),fontsize=14)

plt.show()
