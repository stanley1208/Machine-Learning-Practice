import numpy as np
import matplotlib.pyplot as plt
# dot, trace, det, eig, inv
x=np.array([[1,2,3],[4,5,6]])
y=np.array([[1,2],[4,5],[7,8]])
print(x.dot(y))


position=0
walk=[position]
steps=1000

for i in range(steps):
    step=1 if np.random.randint(0,2) else -1
    position+=step
    walk.append(position)

# 求甚麼時候第一次，距離初始點10步遠
print((np.abs(walk)>10).argmax())
# print(np.abs(walk).argmax())
print(walk)
plt.plot(walk)
plt.show()








