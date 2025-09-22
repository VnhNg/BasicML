import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1, b=0, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.b = b
        self.degree = degree
        self.models = []
        
    def set_params(self, **params):
        """Set parameters for SVM"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def _kernel_function(self, X1, X2):
        gamma = self.gamma
        b = self.b
        degree = self.degree
        """Compute kernel function"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (gamma * (X1 @ X2.T) + b)**degree
        elif self.kernel == 'rbf':
            sq_dist_matrix = np.sum(X1**2, axis=1).reshape(-1, 1) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1).reshape(1, -1)
            return np.exp(-gamma * sq_dist_matrix)
        elif self.kernel == 'sigmoid':
            return np.tanh(gamma * (X1 @ X2.T) + b)
        else:
            raise ValueError("Kernel not recognized")
            
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = X.shape[0]  # n: number of observations
        self.classes_ = np.unique(y)  # Get unique class labels
        gramm_matrix = self._kernel_function(X, X) # Compute gramm matrix   
        
        '''Find boundary hyperplane for each class - OvR'''
        for class_name in self.classes_:
            y_binary = np.where(y == class_name, 1, -1) # Encode the labels: target class vs others
            weights = {'positive': n/(2*np.sum(y_binary == 1) ),
                       'negative': n/(2*np.sum(y_binary == -1) )} # Compute class weight for handling imbalanced datasets
            
            '''Solve dual problem using cvxopt'''
            # Objective function to mininize: sum(np.outer(alphas*y_binary, alphas*y_binary) * gramm_matrix) - sum(alphas)
            P = matrix(np.outer(y_binary, y_binary) * gramm_matrix)
            q = matrix(-np.ones(n))
            
            # Bounds: 0 <= alpha_i <= C_i.  
            G = matrix(np.vstack((-np.eye(n), np.eye(n))))  
            h = matrix(np.hstack((np.zeros(n), np.where(y_binary == 1, 
                                                        self.C * weights['positive'], 
                                                        self.C * weights['negative']))))
            # Constraint: sum(alphas * y_binary) = 0
            A = matrix(y_binary.astype(float), (1, n))  
            b = matrix(np.zeros(1))

            # Solve the QP problem using cvxopt solver
            solution = solvers.qp(P, q, G, h, A, b)
            
            '''Extract parameters'''
            alphas = np.array(solution['x']).flatten()
            sv_indices = np.where(alphas > 0.1)[0] # Set threshold to choose support vectors
            support_vectors = X[sv_indices]
            support_vector_alphas = alphas[sv_indices]
            support_vector_labels = y_binary[sv_indices]
            bias = np.mean(support_vector_labels - 
                                  np.sum(support_vector_alphas * 
                                         support_vector_labels * 
                                         gramm_matrix[np.ix_(sv_indices, sv_indices)], axis=1))
            
            '''Store the model'''
            self.models.append({
                'class_name': class_name,
                'support_vectors': support_vectors,
                'support_vector_alphas': support_vector_alphas,
                'support_vector_labels': support_vector_labels,
                'bias': bias,
            })
            
    def predict(self, X):
        X = np.array(X)
        decision_values = np.zeros((X.shape[0], len(self.models)))
        for idx, model in enumerate(self.models):
            decision_values[:, idx] = np.sum(model['support_vector_alphas'] * 
                                             model['support_vector_labels'] * 
                                             self._kernel_function(X, model['support_vectors']), axis=1) + model['bias'] # Get the distance to each boundary 
        predicted_class_idx = np.argmax(decision_values, axis=1) # Get the class index with the maximum distance 
        predicted_class = self.classes_[predicted_class_idx] # Map the class index to the class label
        return predicted_class
        
    def decision_function(self, X):
        X = np.array(X)
        decision_values = np.zeros((X.shape[0], len(self.models)))
        for idx, model in enumerate(self.models):
            decision_values[:, idx] = np.sum(model['support_vector_alphas'] * 
                                             model['support_vector_labels'] * 
                                             self._kernel_function(X, model['support_vectors']), axis=1) + model['bias'] 
        return decision_values
    

