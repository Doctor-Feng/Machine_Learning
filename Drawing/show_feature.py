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
