# EXP6 - Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## Aim:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries.
  
2. Upload and read the dataset.
   
3. Check for any null values using the isnull() function.
   
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
   
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by :   Tarun S
RegisterNumber:  212223040226
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
# DATASET
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/9c8c3e90-852e-4631-8359-fa6f3d6ec11c)


# data.info()
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/7f49d81b-e01d-4aef-a666-7055e9f2aeeb)


# CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/6411397f-a2df-48f4-8a67-0f7c04432afd)


# VALUE_COUNTS()
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/e77202fb-b69d-45da-880a-370d1f66e3b6)


# DATASET AFTER ENCODING
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/7a441828-7492-43e9-9f78-472262867269)


# X-VALUES
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/b9ef796c-d8a9-4128-9e02-f6c2eea6e0bc)


# ACCURACY
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/666a5564-7fff-4a9a-882e-e9e4e1e951c9)


# dt.predict()
![image](https://github.com/Tarun-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145584190/6e64fa39-e9f2-4b99-b722-f7487069f405)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

