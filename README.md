# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and create/load the employee dataset containing features such as years of experience, monthly working hours, and salary with employee churn status.
2.Split the dataset into training and testing sets.
3.Train the Decision Tree Classifier using the training data and predict the employee churn status for the test data.
4.Evaluate the model using accuracy and confusion matrix, and predict the churn status for a new employee.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R S Kanishka
RegisterNumber: 212225230128 
*/
```


```

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
# Features: [Years of Experience, Monthly Working Hours, Salary (in thousands)]
X = np.array([
    [1, 180, 25],
    [2, 200, 28],
    [3, 210, 30],
    [4, 190, 35],
    [5, 170, 40],
    [6, 160, 45],
    [7, 150, 50],
    [8, 140, 55],
    [9, 130, 60],
    [10, 120, 70]
])

# Target: 0 = Stay, 1 = Churn
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Churn Status:")

print(y_test)

print("\nPredicted Churn Status:")

print(y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# Predict for a new employee
new_employee = np.array([[4, 185, 36]])

prediction = model.predict(new_employee)

if prediction[0] == 1:

    print("\nPrediction: Employee is likely to Churn")
else:

    print("\nPrediction: Employee is likely to Stay")
```


## Output:

<img width="1024" height="765" alt="image" src="https://github.com/user-attachments/assets/44e467e6-c2bd-4f9c-a478-15ef03a3aee1" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
