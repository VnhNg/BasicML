import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def log_loss(y_ohc, prob):
    return -np.sum(y_ohc * np.log(prob + 1e-10)) / y_ohc.shape[0]

class SoftmaxRegression:
    def __init__(self, lrate=0.01, epochs=10000, tol=1e-4, random_seed=42, L1=0, L2=0):
        self.lrate = lrate
        self.epochs = epochs
        self.tol = tol
        self.L1 = L1
        self.L2 = L2
        self.random_seed = random_seed
        self.coef = None
        self.intercept = None
        
    def set_params(self, **params):
        """Set parameters for RidgeRegression"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        y = np.array(y)
        n, m = X.shape  # n: number of observations, m: number of features
        self.classes_, self.number_by_class = np.unique(y, return_counts=True)
        k = len(self.classes_) # k: number of classes
        
        X_b = np.c_[X, np.ones((n, 1))]  # Add intercept term (X_b includes a column of ones for the intercept)
        np.random.seed(self.random_seed)
        W = np.random.randn(m+1, k) # Initialize weights and bias for each class
        
        '''One-hot encode y'''
        y_ohc = np.zeros((n, k))
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        for obs in range(n):
            y_ohc[obs][class_to_index[y[obs]]] = 1 
        
        '''Gradient descent loop'''
        best_loss = float('inf') # Initialize loss for tracking 
        
        for _ in range(self.epochs):
            prob = softmax(X_b.dot(W)) # Compute probabilities
            
            # Track loss to store best parameters
            loss = log_loss(y_ohc, prob)
            if loss < best_loss:
                best_loss = loss
                best_W = W
                best_epoch = _
            
            # Update weights
            gradient = (1/n) * X_b.T.dot(prob - y_ohc) # Compute gradient
            balanced_gradient = gradient * n/(k*self.number_by_class + 0.1) # Balance out 
            W -= self.lrate * balanced_gradient        
            W[:-1,:] -= self.lrate * (self.L2/n) * W[:-1, :] # Add L2 regularization            
            W[:-1,:] = np.sign(W[:-1,:]) * np.maximum(0, np.abs(W[:-1,:]) - self.lrate * self.L1/n) # Add L1 regularization
            
            # Check convergence
            if np.linalg.norm(balanced_gradient) < self.tol: 
                print(f"Converged at epoch {_+1}")
                best_W = W
                break
        else:
            print(f"Not converged yet at epoch {_+1}")
        
        '''Extract parameters'''
        self.intercept = best_W[-1,:]
        self.coef = best_W[:-1,:]
        print(f"best loss {best_loss} at epoch {best_epoch}")
        
    def predict(self, X):
        probabilities = softmax(np.dot(X, self.coef) + self.intercept)  # Get probabilities from softmax    
        predicted_class_idx = np.argmax(probabilities, axis=1) # Get the class index with the maximum probability
        predicted_class = self.classes_[predicted_class_idx] # Map the class index to the class label
        return predicted_class
    
    def predict_proba(self, X):
        return softmax(np.dot(X, self.coef) + self.intercept)
         