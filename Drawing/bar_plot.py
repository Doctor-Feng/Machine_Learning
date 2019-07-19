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

