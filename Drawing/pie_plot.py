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
plt.pie(size,explode = explode ,labels = labels, autopct = '%1.1f%%', shadow = True , startangle =
        90)
plt.axis('equal')
#plt.savefig("origin wind power use percentage diagram.png")

plt.show()

