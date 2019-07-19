# -*- coding: utf-8 -*- 

import matplotlib.pylab as plt
import numpy as np

data1 = np.genfromtxt('001.txt')

# define the rows region
rows = np.arange(871,931)

x = data1[rows,0];#读取了第一列
y1 = data1[rows,1];#读取了第二列

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

plt.plot(x,y1,
         "-",
         label = 'time',
         linewidth = 2)

plt.xlabel("time")
plt.ylabel("value")
plt.title("")

plt.legend(loc = 'upper right')

plt.grid()

plt.show()

#plt.savefig("result.png",dpi = 200)
