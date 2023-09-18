# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: THANJIYAPPAN.K
RegisterNumber:212222240108
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head()
dataset.tail()
dataset = dataset.drop(['sl_no','ssc_p','gender','hsc_s'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset.info()
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset.info()
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

dataset.head()
from sklearn.linear_model import LogisticRegression
#clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
clf = LogisticRegression()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
clf.predict([[0,91,1,58,2,0,55.0,1,58.80]])

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
