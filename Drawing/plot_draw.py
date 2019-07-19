# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

data1 = np.loadtxt("BPiterresult.txt")
data2 = np.loadtxt("result.txt")
data3 = np.loadtxt("GANN0result.txt")

x = data1[:,0]-1;#read the 1st col data
y1 = data1[:,1];#read the 2nd col in data
y2 = data2[:,1];#read the 3rd col in data
y3 = data3[:,1];

#plot setting
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
         label = "BP neural network",
         linewidth = 1)

plt.plot(x,y3,
         "--",
         label = "GANN network",
         linewidth = 2)

plt.plot(x,y2,
         "-",
         label = "optimized GANN by Hill Climbing",
         linewidth = 3)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("")

plt.legend(loc = 'upper right')

plt.grid()

plt.show()

#plt.savefig("result.png",dpi = 200)
