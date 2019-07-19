# draws
functions for drawing
=======
bar_plot
--------
```Python
import numpy as np
import matplotlib.pyplot as plt

plt.figure (1)
index = [0.3,0.8]
plt.bar(index,[0.212,0.002],0.25,alpha = 0.8,color = 'b')
plt.ylabel('time(ms)')
plt.title('')
plt.xticks( np.add(index,0.5 * 0.25),('train','test'))

plt.legend()
#plt.savefig('wind_Power_Usage_Diagram.png',dpi = 600)

plt.show()
```
![](https://github.com/Doctor-Feng/draws/blob/master/bar_plot.png)  

pie_plot
----------
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt')

plt.figure(1)

labels = '0:00-7:00' , '7:00-10:00' ,'10:00-15:00' ,'15:00-18:00' ,'18:00-21:00' ,'21:00-0:00'
explode = (0.1,0,0,0,0,0)
size = [ data[0:27,1].sum() / data[:,1].sum() ,
        data[27:39,1].sum() / data[:,1].sum() ,
        data[39:59,1].sum() / data[:,1].sum() ,
        data[59:71,1].sum() / data[:,1].sum() ,
        data[71:83,1].sum() / data[:,1].sum() ,
        data[83:95,1].sum() / data[:,1].sum()]
plt.pie(size,explode = explode ,labels = labels, autopct = '%1.1f%%', shadow = True , startangle = 90)
plt.axis('equal')
#plt.savefig("origin wind power use percentage diagram.png")

plt.show()
```
![](https://github.com/Doctor-Feng/draws/blob/master/pie_plot.png)

plot_draw
--------
```python
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
```
show_feature
------------
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def list_generator(mean,dis,number):
    return np.random.normal(mean,dis*dis,number)
data = np.loadtxt('data.txt')
list1=data[:,0]
list2=data[:,1]
list3=data[:,2]
data = pd.DataFrame({"Load":list1,
                     "Wind":list2,
                     "Light":list3
                    })
data.boxplot()
plt.ylabel("Infomation")
plt.xlabel("Kinds of Power")
plt.show()
```
![](https://github.com/Doctor-Feng/draws/blob/master/show_feature.png)

