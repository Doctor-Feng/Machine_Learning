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

