import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from matplotlib import gridspec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split




def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = math.ceil((len(instances)/images_per_row))

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image,  **options)
    plt.axis("off")


mnist=fetch_openml('mnist_784',version=1,as_frame=False)
mnist.target=mnist.target.astype(np.uint8)

X=mnist["data"]
y=mnist["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y)

print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

pca=PCA()
pca.fit(X_train)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>=0.95)+1
print(cumsum)
print(d)

pca=PCA(n_components=0.95)
X_reduced=pca.fit_transform(X_train)
print(pca.n_components_)
print(np.sum(pca.explained_variance_ratio_))


# plt.figure(figsize=(10,8))
# plt.plot(cumsum,linewidth=3)
# plt.axis([0,400,0,1])
# plt.xlabel("Dimensions")
# plt.ylabel("Explained Variance")
# plt.plot([d,d],[0,0.95],"k:")
# plt.plot([0,d],[0.95,0.95],"k:")
# plt.plot(d,0.95,"ko")
# plt.annotate("Elbow",xy=(65,0.85),xytext=(70,0.7),arrowprops=dict(arrowstyle="->"),fontsize=18)
# plt.grid(True)
# plt.show()


pca=PCA(n_components=154)
X_reduced=pca.fit_transform(X_train)
X_recovered=pca.inverse_transform(X_reduced)

X_reduced_pca=X_reduced

plt.figure(figsize=(7,4))
plt.subplot(121)
plot_digits(X_train[::21000])
plt.title("Original",fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::21000])
plt.title("Compressed",fontsize=16)
plt.show()


rnd_pca=PCA(n_components=154,svd_solver='randomized',random_state=42)
X_reduced=rnd_pca.fit_transform(X_train)


n_batches=100
# inc_pca=IncrementalPCA(n_components=154)
# for X_batch in np.array_split(X_train,n_batches):
#     print(".",end='')
#     inc_pca.partial_fit(X_batch)


# X_reduced=inc_pca.fit_transform(X_train)
#
# X_reduced_inc_pca=X_reduced

# print()
# print(np.allclose(pca.mean_,inc_pca.mean_))
# print(np.allclose(X_reduced_pca,X_reduced_inc_pca))


filename="my_mnist.data"
m,n=X_train.shape

X_mm=np.memmap(filename,dtype='float32',mode='write',shape=(m,n))
X_mm[:]=X_train

del X_mm

X_mm=np.memmap(filename,dtype='float32',mode='readonly',shape=(m,n))

batch_size=m//n_batches
inc_pca=IncrementalPCA(n_components=154,batch_size=batch_size)
print(inc_pca.fit(X_mm))


for n_components in (2,10,154):
    print("n_components :",n_components)
    regular_pca=PCA(n_components=n_components,svd_solver='full')
    inc_pca=IncrementalPCA(n_components=n_components,batch_size=500)
    rnd_pca=PCA(n_components=n_components,random_state=42,svd_solver="randomized")
    for name,pca in (("PCA",regular_pca),("Incremental PCA",inc_pca),("Random PCA",regular_pca)):
        t1=time.time()
        pca.fit(X_train)
        t2=time.time()
        print("{}:{:.2f} seconds".format(name,t2-t1))



