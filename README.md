# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import NumPy and initialize the input dataset X and target values y.
2. Normalize the feature values using mean and standard deviation.
3. Add the bias term, initialize weights, learning rate, and epochs.
4. Train the model by predicting outputs, calculating error, and updating weights using gradient descent.
5. Print the trained weights and predicted output values.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: NITHISH R
RegisterNumber: 212225230202
*/

import numpy as np

X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

y = np.array([50, 45, 70, 80, 95], dtype=float)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X = np.c_[np.ones(X.shape[0]), X]

n_features = X.shape[1]
weights = np.zeros(n_features)

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)
```

## Output:


<img width="893" height="55" alt="ML EX 4" src="https://github.com/user-attachments/assets/c022ee03-96b6-4710-8f4a-0ee05f69d8a6" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
