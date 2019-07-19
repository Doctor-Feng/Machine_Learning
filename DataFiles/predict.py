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
