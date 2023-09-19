# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student . 

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
### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
```

# Developed by: THANJIYAPPAN.K
# RegisterNumber: 212222240108
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])

```

## Output:
### read csv file:
![1](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/9c06188f-f8da-405f-b95c-0adb4a366ce8)


### to read first ten data(head):
![2](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/0954b6aa-e13f-4b6d-9385-ebdfe4ef2055)

### to read last ten data(tail):
![3](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/5a4da338-e1e9-4b55-8529-ff983b29872f)


### Dropping the serial number and salary column:
![4](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/57897919-67dc-4f76-96c0-029ae36643e8)

### Dataset Shape:
![5](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/2b153f1c-bf3d-4046-a900-7e9698ccd2f3)

### Dataset Information:
![6](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/1b75660d-a6db-4579-bb03-92ca1b96c8d8)

### Dataset after changing object into category:
![7](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/9bb9632a-f36e-45e0-972c-145ad7b1a1c5)

### Dataset after changing category into integer:
![8](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/d862c257-d3eb-4bb9-955e-fd875fba5c5e)

### Displaying the Dataset:
![9](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/bf84e5e1-12db-490f-9aed-195aa558324e)

### Selecting the features and labels:
![10](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/551ad914-00d0-4587-ac24-4a97dc61bed4)

### Dividing the data into train and test:
![11](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/c7608042-0339-4ae3-b06e-7b6038f7c7e6)
### Shape of x_train and y_train:
![12](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/c45cb545-cbaf-4854-b1b9-66c438940544)
### Creating a Classifier using Sklearn:
![13](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/fcd601ce-f052-40bd-acf3-16fd2a7ecb34)

Predicting for random value:
![14](https://github.com/22009011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343461/4dcc1e0c-3190-4f6d-b407-b66b80f5224a)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
