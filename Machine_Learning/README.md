# Machine_Learning
Bayes
===============
Code for Bayes
--------------
```python

#!/usr/bin/python

from sklearn import datasets
iris = datasets.load_iris()

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(iris.data,
                                                iris.target,test_size = 0.2,random_state = 0)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report
print classification_report(y_test,y_pred)
```
output for Bayes
-------------


             precision    recall  f1-score   support

          0       1.00      1.00      1.00        11
          1       0.93      1.00      0.96        13
          2       1.00      0.83      0.91         6

avg / total       0.97      0.97      0.97        30

