# ./simpliest.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
data = np.loadtxt("testSet.txt")
train_x = data[0:80,0:-1]
train_y = data[0:80,-1]
test_x = data[80:100,0:-1]
test_y = data[80:100,-1]
start_time = time.time()
model = svm.LinearSVC()
model.fit(train_x, train_y)
print 'Training took %fs!' % (time.time() - start_time)
predict = model.predict(test_x)
accuracy = accuracy_score(test_y, predict)
print 'predict took %fs!' % ((time.time() - start_time)/ 20)
print 'Accuracy: %.2f%%' % (100 * accuracy)
```
# ./train.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn import linear_model
from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import os.path
import glob
import cv2
dataSet = []
output = []
svm = svm.LinearSVC()

def jpg_data():
    train_time = 0
    for jpgs in glob.glob("./cat_dogs/*.thumbnail.jpg"):
        data = cv2.imread(jpgs,0).ravel().tolist()
        if jpgs.split('/')[2].startswith("cat"):
            out = -1
        if jpgs.split('/')[2].startswith("dog"):
            out = 1
        train_time = train_time + 1
        if train_time == 2000:
            break
        print "the traintime is %d" % train_time
        dataSet.append(data)
        output.append(out)
if __name__ == "__main__":
    print "#Preparing for collecting data..."
    jpg_data()
    print '#Preparing for traing model'
    svm.fit(dataSet, output)
    print '# train model is done..'
    joblib.dump(svm, "train_model.m")


```
# ./test.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
data = np.loadtxt("testSet.txt")
train_x = data[0:80,0:-1]
train_y = data[0:80,-1]
test_x = data[80:100,0:-1]
test_y = data[80:100,-1]
num_train, num_feat = train_x.shape
num_test, num_feat = test_x.shape
print '#Training data: %d, #Testing_data: %d, Dimension: %d' % (num_train, num_test, num_feat)
print '*******************SVM Linear Kernel Classifier********************'
start_time = time.time()
model = svm.LinearSVC()
model.fit(train_x, train_y)
print 'Training took %fs!' % (time.time() - start_time)
test_time = time.time()
predict_array = []
for i in range(20):
    predict_array.append(test_x[i].reshape(1,-1))
for i in range(20):
    print "the actual is:",test_y[i], "the predict is:",model.predict(predict_array[i])
print 'Test time took %fs!' % ((time.time() - test_time)/20)
predict = model.predict(test_x)
accuracy = accuracy_score(test_y, predict)
print 'Accuracy: %.2f%%' % (100 * accuracy)
```
# ./resize.py
------
```python
#!/usr/bin/python
import os, glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
path = raw_input("path:")
width =int(raw_input("the width U want:"))
height =int(raw_input("the height U want:"))
imgslist = glob.glob(path+'/*.*')
#format = raw_input("format:")
format = "jpg"
def small_img():
    for imgs in imgslist:
        imgspath, ext = os.path.splitext(imgs)
        img = Image.open(imgs)
        (x,y) = img.size
        #height =int( y * width /x )
        small_img =img.resize((width,height),Image.ANTIALIAS)
        small_img.save(imgspath +".thumbnail."+format)
        print "done"
if __name__ == '__main__':
    small_img()

```
# ./study/train.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn import linear_model
from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import os.path
import glob
import cv2
dataSet = []
output = []
svm = svm.SVC()
def jpg_data():
    train_time = 0
    for jpgs in glob.glob("./study_images/train_images/*.jpg"):
        data = cv2.imread(jpgs,0).ravel().tolist()
        if jpgs.split('/')[3].startswith("with_catter"):
            out = 1
        if jpgs.split('/')[3].startswith("no_catter"):
            out = -1
        train_time = train_time + 1
        if train_time == 2000:
            break
        print "the traintime is %d" % train_time
        dataSet.append(data)
        output.append(out)
if __name__ == "__main__":
    print "#Preparing for collecting data..."
    jpg_data()
    start_time = time.time()    
    print '#Preparing for traing model'
    svm.fit(dataSet, output)
    print "program run time for ", time.time() - start_time
    print '# train model is done..'
    joblib.dump(svm, "train_model.m")
```
# ./study/resize.py
------
```python
#!/usr/bin/python
import os, glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
path = raw_input("path:")
width =int(raw_input("the width U want:"))
height =int(raw_input("the height U want:"))
imgslist = glob.glob(path+'/*.*')
#format = raw_input("format:")
format = "jpg"
def small_img():
    for imgs in imgslist:
        imgspath, ext = os.path.splitext(imgs)
        img = Image.open(imgs)
        (x,y) = img.size
        #height =int( y * width /x )
        small_img =img.resize((width,height),Image.ANTIALIAS)
        small_img.save(imgspath + "." + format)
        print "done"
if __name__ == '__main__':
    small_img()

```
# ./study/predict.py
------
```python
#!/usr/bin/python
import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import glob
import time

predict_num = 5
model = joblib.load("train_model.m")
dataSet = []
output = []

def calc_accuracy():
    start_time = time.time()
    predict = model.predict(dataSet)
    for i in range(predict_num):
        print "the actual is:",output[i], "the predict is:",predict[i]
    accuracy = accuracy_score(output, predict)
    print 'Accuracy: %.2f%%' % (100 * accuracy)
    print "average time for one photo is ", (time.time() - start_time)/ 5
def jpg_data():
    train_time = 0
    for jpgs in glob.glob("./study_images/predict_images/*.jpg"):
        data = cv2.imread(jpgs,0).ravel().tolist()
        if jpgs.split('/')[3].startswith("no_catter"):
            out = -1
        if jpgs.split('/')[3].startswith("with_catter"):
            out = 1
        train_time = train_time + 1
        if train_time == 100:
            break
        print "the traintime is %d" % train_time
	print jpgs.split('/')[3]
        dataSet.append(data)
        output.append(out)

if __name__ == "__main__":
    jpg_data()
    calc_accuracy()
```
# ./study/bar_plot.py
------
```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure (1)
index = [0.3,0.8]
plt.bar(index,[0.207,0.114],0.25,alpha = 0.8,color = 'b')
plt.ylabel('time(ms)')
plt.title('')
plt.xticks( np.add(index,0.5 * 0.25),('train','test'))

plt.legend()
#plt.savefig('wind_Power_Usage_Diagram.png',dpi = 600)

plt.show()

```
# ./plot_svm_regression.py
------
```python
"""
===================================================================
Support Vector Regression (SVR) using linear and non-linear kernels
===================================================================

Toy example of 1D regression using linear, polynomial and RBF kernels.

"""
print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

###############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
```
# ./force.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import numpy as np

data = np.loadtxt("data.txt",delimiter=",")
train_x = data[0:60,0:-1]
train_y = data[0:60,-1]
test_x = data[20:30,0:-1]
test_y = data[20:30,-1]
num_train, num_feat = train_x.shape
num_test, num_feat = test_x.shape
print '#Training data: %d, #Testing_data: %d, Dimension: %d' % (num_train, num_test, num_feat)
print '*******************SVM Linear Kernel Regression********************'
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                    "gamma":np.logspace(-2, 2, 5)})
start_time = time.time()
svr.fit(train_x, train_y)
print "SVR complexity and bandwidth selected and model fitted in %.3f s" % (time.time()-start_time)
sv_ratio = svr.best_estimator_.support_.shape[0] / 100
print("Support vector ratio: %.3f" % sv_ratio)
for i in range(10):
    print "the actual is:",test_y[i], "the predict is:",svr.predict(test_x[i].reshape(1,-1))

```
# ./svm.py
------
```python
#!/usr/bin/python
import time
import sklearn
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
data = np.loadtxt("testSet.txt")
train_x = data[0:80,0:-1]
train_y = data[0:80,-1]
test_x = data[80:100,0:-1]
test_y = data[80:100,-1]
start_time = time.time()
model = svm.LinearSVC()
model.fit(train_x, train_y)
print 'Training took %fs!' % (time.time() - start_time)
predict = model.predict(test_x)
accuracy = accuracy_score(test_y, predict)
print 'Accuracy: %.2f%%' % (100 * accuracy)
```
# ./predict.py
------
```python
#!/usr/bin/python
import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import glob

model = joblib.load("train_model.m")
dataSet = []
output = []

def predict_one():
    #predict
    jpg = './cat_dogs/cat.999.thumbnail.jpg'
    print jpg
    data = cv2.imread(jpg,0).reshape(1,-1)
    if model.predict(data) == 1:
        print "the result of predict is a dog!"
    else:
        print "the result of predict is a cat!"

def calc_accuracy():
    predict = model.predict(dataSet)
    for i in range(99):
        print "the actual is:",output[i], "the predict is:",predict[i]
    accuracy = accuracy_score(output, predict)
    print 'Accuracy: %.2f%%' % (100 * accuracy)

def jpg_data():
    train_time = 0
    for jpgs in glob.glob("./cat_dogs/*.thumbnail.jpg"):
        data = cv2.imread(jpgs,0).ravel().tolist()
        if jpgs.split('/')[2].startswith("cat"):
            out = -1
        if jpgs.split('/')[2].startswith("dog"):
            out = 1
        train_time = train_time + 1
        if train_time == 100:
            break
        print "the traintime is %d" % train_time
        dataSet.append(data)
        output.append(out)

if __name__ == "__main__":
    jpg_data()
    calc_accuracy()
```
