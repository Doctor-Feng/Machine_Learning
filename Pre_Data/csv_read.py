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
