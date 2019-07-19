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
