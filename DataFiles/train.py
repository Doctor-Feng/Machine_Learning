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


