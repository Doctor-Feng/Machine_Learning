# ./csv_read.py
------
```python
import pandas as pd
filename = "1.csv"
#=====================================Read small data=====================================
data = pd.read_csv(filename,
                   #if the data
                   #header = None ,
                   #if the file has no names, it will autoname it, otherwise
                   #names = ['fields1','fields2']
                   #if has the date and time, it will auto detect it
                   #if the data contain bad data that columns number cannot parallel with others
                   # error_bad_lines = False,
                   #if the data contain index field
                    index_col = 0
                   )
#=====================================Big data============================================
# if the data is too big, more than 5M
# you should use the chunk
#  data = pd.read_csv(filename,
#                    header = None,
#                     names = ['C1','C2','C3','C4','C5']
#                     chunksize = 10
#                     )
#  for chunk in data.chunks:
#      print chunk



print data.info()

#if the data contain NAN namely missing data
# it will replace the missing data with 50

#data.fillna(50)
#data.fillna(-1)

#it will replace it with the mean value
#data.fillna(data.mean(axis=0))

#if u want to change the origin data
#data.fillna(50, inplace = True)

#==========================================Data plit===============================================
print data.iloc[4,2]
print data.iloc[:,0]
print data['X Location (mm)'] # get a column of X Location

#==========================================Process Label data=========================
# we need change this string into [1,0,0,0] binary value
#  text_field = data['weather']
#  text_label  = np.unique(text_field)
#  text_label = pd.Series(text_label)
#  label_binary = pd.get_dummies(text_label)
#
#  # we can see the text change into binary vector
#  label_binary['sunny']
```
# ./txt_read.py
------
```python
# -*- coding: utf-8 -*- 

import matplotlib.pylab as plt
import numpy as np

data1 = np.genfromtxt('11.txt',delimiter=' ')[:,1:]

x = data1[:,0]-1;#读取了第一列
y1 = data1[:,1];#读取了第二列

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
```
# ./solve_xlsx.py
------
```python
import xlrd

book = xlrd.open_workbook('cmy.xlsx')

for sheet in book.sheets():
    print sheet.name

sheet = book.sheet_by_name('Sheet1')

print sheet.nrows

BsmI = []
AapI = []
FokI = []
TaqI = []
Cdx2 = []


for i in range(sheet.nrows):
    print sheet.row_values(i)
    if 'BsmI' in sheet.row_values(i)[1]:
        BsmI.append( sheet.row_values(i)[0])
    if 'AapI' in sheet.row_values(i)[1]:
        AapI.append( sheet.row_values(i)[0])
    if 'FokI' in sheet.row_values(i)[1]:
        FokI.append( sheet.row_values(i)[0])
    if 'TaqI' in sheet.row_values(i)[1]:
        TaqI.append( sheet.row_values(i)[0])
    if 'Cdx2' in sheet.row_values(i)[1]:
        Cdx2.append( sheet.row_values(i)[0])

print 'BsmI'
print BsmI
print '=' * 20

print 'AapI'
print AapI
print '=' * 20

print 'FokI'
print FokI
print '=' * 20

print 'TaqI'
print TaqI
print '=' * 20

print 'Cdx2'
print Cdx2
print '=' * 20


```
