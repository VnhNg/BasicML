import numpy as np

class RidgeRegression:

    def __init__(self, alpha=0.1, lrate=0.01, epochs=10000, tol=1e-4, random_seed=None):
        self.alpha = alpha # Regularization of weights
        self.lrate = lrate # Learning rate        
        self.epochs = epochs # Maximal number of iratation    
        self.tol = tol # Convergence confirmation threshold
        self.random_seed = random_seed # Set seed to generate paramters
        self.coef = None
        self.intercept = None
        
    def set_params(self, **params):
        """Set parameters for RidgeRegression"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        n, m = X.shape # n: number of observations, m: number of features
        
        X_b = np.c_[X, np.ones((n, 1))] # Add intercept term 
        if self.random_seed != None:
            np.random.seed(self.random_seed)
            W = np.random.randn(m+1) # Initialize weights
        else: W = np.zeros(m+1)
                
        '''Gradient descent loop'''
        best_loss = float('inf') # Initialize loss for tracking
        
        for _ in range(self.epochs):
            errors = y - X_b.dot(W) 
            
            # Track loss to store best parameters
            loss = np.mean(errors**2) + self.alpha/n * np.sum(W**2)
            if loss < best_loss:
                best_loss = loss
                best_W = W
                best_epoch = _
            
            # Update weights
            gradient = (-1/n) * X_b.T.dot(errors) + (self.alpha/n) * np.r_[W[:-1], np.zeros(1)]
            W -= self.lrate * gradient 
            
            # Check convergence
            if np.linalg.norm(gradient) < self.tol: 
                print(f"Converged at epoch {_+1}")
                best_W = W
                break
        else:
            print(f"Not converged yet at epoch {_+1}")
        
        '''Extract parameters'''
        self.intercept = best_W[-1]
        self.coef = best_W[:-1]
        print(f"Till epoch {_} best loss {best_loss} at epoch {best_epoch}")

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
    

