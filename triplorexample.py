# -*- coding: utf-8 _*_
# @Time : 8/10/2021 10:08 am
# @Author: ZHA Mengyue
# @FileName: triplorexample.py
# @Software: Blackjack
# @Blog: https://github.com/Dolores2333


# Import libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Creating radii and angles
r = np.linspace(0.125, 1.0, 100)
a = np.linspace(0, 2 * np.pi,
                100,
                endpoint=False)

# Repeating all angles for every radius
a = np.repeat(a[..., np.newaxis], 100, axis=1)

# Creating datset
x = np.append(0, (r * np.cos(a)))
y = np.append(0, (r * np.sin(a)))
z = (np.sin(x ** 4) + np.cos(y ** 4))

# Creating figure
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection='3d')

# Creating color map
my_cmap = plt.get_cmap('hot')

# Creating plot
trisurf = ax.plot_trisurf(x, y, z,
                          cmap=my_cmap,
                          linewidth=0.2,
                          antialiased=True,
                          edgecolor='grey')
fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
ax.set_title('Pseudo Value Function Plot')

# Adding labels
ax.set_xlabel('dealer', fontweight='bold')
ax.set_ylabel('player', fontweight='bold')
ax.set_zlabel('value', fontweight='bold')

# show plot
plt.savefig('axis_example.png')
plt.show()