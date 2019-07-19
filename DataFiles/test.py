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
