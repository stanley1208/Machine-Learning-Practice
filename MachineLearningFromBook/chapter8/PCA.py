import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from matplotlib import gridspec


class Arrow3D(FancyArrowPatch):
    def __init__(self,xs,ys,zs,*args,**kwargs):
        FancyArrowPatch.__init__(self,(0,0),(0,0),*args,**kwargs)
        self._verts3d=xs,ys,zs
    def draw(self,renderer):
        xs3d,ys3d,zs3d=self._verts3d
        xs,ys,zs=proj3d.proj_transform(xs3d,ys3d,zs3d,renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self,renderer)



np.random.seed(4)
m=60
w1,w2=0.1,0.3
noise=0.1


angles=np.random.rand(m)*3*np.pi/2-0.5
# print(angles)
X=np.empty((m,3))
X[:,0]=np.cos(angles)+np.sin(angles)/2+noise*np.random.randn(m)/2
X[:,1]=np.sin(angles)*0.7+noise*np.random.randn(m)/2
X[:,2]=X[:,0]*w1+X[:,1]*w2+noise+np.random.randn(m)

X_centered=X-X.mean(axis=0)
U,s,Vt=np.linalg.svd(X_centered)
c1=Vt.T[:,0]
c2=Vt.T[:,1]

print(c1)
print(c2)

m,n=X.shape
S=np.zeros(X_centered.shape)
S[:n,:n]=np.diag(s)

print('s',s)
print('S',S)
print(np.allclose(X_centered,U.dot(S).dot(Vt)))

W2=Vt.T[:,:2]
X2D=X_centered.dot(W2)
X2D_using_svd=X2D


pca=PCA(n_components=2)
X2D=pca.fit_transform(X)

print(X2D[:5])
print(X2D_using_svd[:5])

print(np.allclose(X2D,-X2D_using_svd))


X3D_inv=pca.inverse_transform(X2D)
print(np.allclose(X3D_inv,X))

print(np.mean(np.sum(np.square(X3D_inv-X),axis=1)))

X3D_inv_using_svd=X2D_using_svd.dot(Vt[:2,:])

# print(X3D_inv_using_svd.shape)
# print((X3D_inv-pca.mean_).shape)
# print(X3D_inv.shape)

print(np.allclose(X3D_inv_using_svd,X3D_inv-pca.mean_))

print(pca.components_)

print(Vt[:2])

print(pca.explained_variance_ratio_)

print(1-pca.explained_variance_ratio_.sum())

print(np.square(s)/np.square(s).sum())



# axes=[-1.8,1.8,-1.3,1.3,-1.0,1.0]
#
# x1s=np.linspace(axes[0],axes[1],10)
# x2s=np.linspace(axes[2],axes[3],10)
# x1,x2=np.meshgrid(x1s,x2s)
#
# C=pca.components_
# R=C.T.dot(C)
# z=(R[0,2]*x1+R[1,2]*x2)/(1-R[2,2])
#
#
# fig=plt.figure(figsize=(10,8))
# ax=fig.add_subplot(111,projection='3d')
#
# X3D_above=X[X[:,2]>X3D_inv[:,2]]
# X3D_below=X[X[:,2]<=X3D_inv[:,2]]
#
# ax.plot(X3D_below[:,0],X3D_below[:,1],X3D_below[:,2],'bo',alpha=0.5)
#
# ax.plot_surface(x1,x2,z,alpha=0.2,color='k')
# np.linalg.norm(C,axis=0)
# ax.add_artist(Arrow3D([0,C[0,0]],[0,C[0,1]],[0,C[0,2]],mutation_scale=15,lw=1,arrowstyle="-|>",color="k"))
# ax.add_artist(Arrow3D([0,C[1,0]],[0,C[1,1]],[0,C[1,2]],mutation_scale=15,lw=1,arrowstyle="-|>",color="#f70505"))
# ax.plot([0],[0],[0],"k.")
#
# for i in range(m):
#     if X[i, 2] > X3D_inv[i, 2]:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
#     else:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#f70505")
#
#
# ax.plot(X3D_inv[:,0],X3D_inv[:,1],X3D_inv[:,2],"k.")
# ax.plot(X3D_above[:,0],X3D_above[:,1],X3D_above[:,2],'bo',alpha=0.8)
#
# ax.set_xlabel("$x_1$",fontsize=18,labelpad=10)
# ax.set_ylabel("$y_1$",fontsize=18,labelpad=10)
# ax.set_zlabel("$z_1$",fontsize=18,labelpad=10)
#
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# plt.show()


# print(X2D)
#
# fig2=plt.figure()
# ax2=fig2.add_subplot(111,aspect='equal')
#
# ax2.plot(X2D[:,0],X2D[:,1],"k+")
# ax2.plot(X2D[:,0],X2D[:,1],"k.")
# ax2.plot([0],[0],"ko")
# ax2.arrow(0,0,0,1,head_width=0.05,length_includes_head=True,head_length=0.1,fc='k',ec='k')
# ax2.arrow(0,0,1,0,head_width=0.05,length_includes_head=True,head_length=0.2,fc='k',ec='k')
# ax2.set_xlabel("$z_1$",fontsize=18)
# ax2.set_ylabel("$z_2$",fontsize=18,rotation=0)
# ax2.axis([-1.5,1.5,-1.2,1.2])
# ax2.grid(True)
# plt.show()


X,t=make_swiss_roll(n_samples=1000,noise=0.5,random_state=42)

print("X: ",X)
print("t: ",t)

# axes = [-11, 14, -2, 23, -12, 15]
#
#
# fig3=plt.figure(figsize=(10,8))
# ax3=fig3.add_subplot(111,projection='3d')
#
# ax3.scatter(X[:,0],X[:,1],X[:,2],c=t,cmap=plt.cm.hot)
# ax3.view_init(10,-70)
# ax3.set_xlabel("$x_1$",fontsize=18)
# ax3.set_ylabel("$x_2$",fontsize=18)
# ax3.set_zlabel("$x_3$",fontsize=18)
# ax3.set_xlim(axes[0:2])
# ax3.set_ylim(axes[2:4])
# ax3.set_zlim(axes[4:6])
#
# plt.show()


# plt.figure(figsize=(10,5))
#
#
# # projecting onto a plane
# plt.subplot(121)
# plt.scatter(X[:,0],X[:,1],c=t,cmap=plt.cm.hot)
# plt.axis(axes[:4])
# plt.xlabel("$x_1$",fontsize=18)
# plt.ylabel("$x_2$",fontsize=18,rotation=0)
# plt.grid(True)
#
# # unrolling the Swiss roll
# plt.subplot(122)
# plt.scatter(t,X[:,1],c=t,cmap=plt.cm.hot)
# plt.axis([4,18,axes[2],axes[3]])
# plt.xlabel("$z_1$",fontsize=18)
# plt.grid(True)
#
# plt.show()



axes=[-11.5,14,23,-2,-12,15]


x2s=np.linspace(axes[2],axes[3],10)
x3s=np.linspace(axes[4],axes[5],10)
x2,x3=np.meshgrid(x2s,x3s)

fig4=plt.figure(figsize=(8,8))
ax4=plt.subplot(111,projection='3d')

positive_class=X[:,0]>5
X_pos=X[positive_class]
X_neg=X[~positive_class]
ax4.view_init(10,-70)
ax4.plot(X_neg[:,0],X_neg[:,1],X_neg[:,2],"y^")
ax4.plot_wireframe(5,x2,x3,alpha=0.5)
ax4.plot(X_pos[:,0],X_pos[:,1],X_pos[:,2],"gs")
ax4.set_xlabel("$x_1$",fontsize=18)
ax4.set_ylabel("$x_2$",fontsize=18)
ax4.set_zlabel("$x_3$",fontsize=18)
ax4.set_xlim(axes[0:2])
ax4.set_ylim(axes[2:4])
ax4.set_zlim(axes[4:6])

# plt.show()

fig5=plt.figure(figsize=(8,8))
ax5=plt.subplot(111)
plt.plot(t[positive_class],X[positive_class,1],'gs')
plt.plot(t[~positive_class],X[~positive_class,1],'y^')
plt.axis([4,15,axes[2],axes[3]])
plt.xlabel("$z_1$",fontsize=18)
plt.ylabel("$z_2$",fontsize=18)
plt.grid(True)

# plt.show()

fig6=plt.figure(figsize=(8,8))
ax6=plt.subplot(111,projection='3d')

positive_class2=2*(t[:]-4)>X[:,1]
X_pos2=X[positive_class2]
X_neg2=X[~positive_class2]
ax6.view_init(10,-70)
ax6.plot(X_neg2[:,0],X_neg2[:,1],X_neg2[:,2],"y^")
ax6.plot(X_pos2[:,0],X_pos2[:,1],X_pos2[:,2],"gs")
ax6.set_xlabel("$x_1$",fontsize=18)
ax6.set_ylabel("$x_2$",fontsize=18)
ax6.set_zlabel("$x_3$",fontsize=18)
ax4.set_xlim(axes[0:2])
ax4.set_ylim(axes[2:4])
ax4.set_zlim(axes[4:6])

# plt.show()


fig7=plt.figure(figsize=(8,8))
ax7=plt.subplot(111)
plt.plot(t[positive_class],X[positive_class,1],'gs')
plt.plot(t[~positive_class],X[~positive_class,1],'y^')
plt.plot([4,15],[0,22],"b-",linewidth=2)
plt.axis([4,15,axes[2],axes[3]])
plt.xlabel("$z_1$",fontsize=18)
plt.ylabel("$z_2$",fontsize=18)
plt.grid(True)

# plt.show()


angle=np.pi/5
stretch=5
m=200

np.random.seed(3)
X=np.random.randn(m,2)/10
X=X.dot(np.array([[stretch,0],[0,1]]))  # stretch the matrix
X=X.dot([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) # rotate the matrix

u1=np.array([np.cos(angle),np.sin(angle)])
u2=np.array([np.cos(angle-2*np.pi/6),np.sin(angle-2*np.pi/6)])
u3=np.array([np.cos(angle-np.pi/2),np.sin(angle-np.pi/2)])

X_proj1=X.dot(u1.reshape((-1,1)))
X_proj2=X.dot(u2.reshape((-1,1)))
X_proj3=X.dot(u3.reshape((-1,1)))

plt.figure(figsize=(8,4))
plt.subplot2grid((3,2),(0,0),rowspan=3)
plt.plot([-1.4,1.4],[-1.4*u1[1]/u1[0],1.4*u1[1]/u1[0]],'k-',linewidth=1)
plt.plot([-1.4,1.4],[-1.4*u2[1]/u2[0],1.4*u2[1]/u2[0]],'k--',linewidth=1)
plt.plot([-1.4,1.4],[-1.4*u3[1]/u3[0],1.4*u3[1]/u3[0]],'k:',linewidth=1)
plt.plot(X[:,0],X[:,1],"bo",alpha=0.8)
plt.axis([-1.4,1.4,-1.4,1.4])
plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)


plt.subplot2grid((3,2),(0,1))
plt.plot([-2,2],[0,0],'k-',linewidth=1)
plt.plot(X_proj1[:,0],np.zeros(m),"bo",alpha=0.5)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2,2,-1,1])
plt.grid(True)

plt.subplot2grid((3,2),(1,1))
plt.plot([-2,2],[0,0],'k-',linewidth=1)
plt.plot(X_proj2[:,0],np.zeros(m),"bo",alpha=0.5)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2,2,-1,1])
plt.grid(True)

plt.subplot2grid((3,2),(2,1))
plt.plot([-2,2],[0,0],'k-',linewidth=1)
plt.plot(X_proj3[:,0],np.zeros(m),"bo",alpha=0.5)
plt.gca().get_yaxis().set_ticks([])
# plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2,2,-1,1])
plt.xlabel("$z_1$",fontsize=18)
plt.grid(True)


plt.show()