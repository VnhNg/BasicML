import numpy as np 

class LinearRegression:   
    
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # Add intercept term 
        X_b = np.c_[X, np.ones((X.shape[0], 1))]
        # (X_b^T * X_b)^(-1) * X_b^T * y
        W = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = W[-1]
        self.coef = W[:-1]

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
        


