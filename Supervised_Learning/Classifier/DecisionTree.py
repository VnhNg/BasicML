import numpy as np

class Node:
    """A node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, n_samples=None):
        self.feature = feature            # Feature index to split on
        self.threshold = threshold        # Split threshold
        self.left = left                  # Left child
        self.right = right                # Right child
        self.value = value                # Predicted class index at this node 
        self.n_samples = n_samples        # Numpy array of class counts for this node
        self.impurity = 0.0               # Node impurity (Gini or Entropy)

    def is_leaf(self) -> bool:
        return (self.left is None) and (self.right is None)


class DecisionTree:
    """
    Decision Tree Classifier with cost-complexity pruning and class balancing.

    Parameters
    ----------
    criterion : str, default='gini'
        Splitting criterion ('gini' or 'entropy').
    random_state : int, default=42
        Random seed for reproducibility.
    splitter : str, default='best'
        Strategy to choose split ('best' or 'random').
    max_features : int or None, default=None
        Number of features to consider when splitting.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a split.
    ccp_alpha : float, default=0.0
        Complexity parameter for pruning. Larger values → more pruning.
    """

    def __init__(self,
                 criterion='gini', random_state=42,
                 splitter='best', max_features=None,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):  
        self.criterion = criterion
        self.random_state = random_state
        self.splitter = splitter
        self.max_features = max_features

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        
        self.root = None
        
        np.random.seed(random_state)
    
    def set_params(self, **params):
        """Set parameters for DecisionTree"""
        for key, value in params.items():
            setattr(self, key, value)
        return self 
    
    def get_params(self, deep=True):
        return {
            "criterion": self.criterion,
            "random_state": self.random_state,
            "splitter": self.splitter,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "ccp_alpha": self.ccp_alpha,
        }

    # ----------------------------
    # Fit
    # ----------------------------
    def _calculate_impurity(self, counts: np.ndarray) -> float:    
        weighted_counts = counts * self.class_weights
        total = weighted_counts.sum()
        if total == 0:
            return 0.0
        proportions = weighted_counts / total
        if self.criterion == 'gini':
            return 1.0 - np.sum(proportions ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(proportions * np.log2(proportions, where=proportions > 0))
        else:
            raise ValueError("Unknown criterion")

    def _split(self, X_column: np.ndarray, threshold: float):
        left_idxs = np.nonzero(X_column <= threshold)[0]
        right_idxs = np.nonzero(X_column > threshold)[0]
        return left_idxs, right_idxs

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_impurity: float):
        best_gain = 0.0
        best_feature, best_threshold = None, None
        total_samples = len(y)

        # Find best split
        features = np.arange(self.n_features)
        if self.max_features is not None and 0 < self.max_features < self.n_features:
            features = np.random.choice(features, self.max_features, replace=False)

        for feature in features:
            values = np.sort(np.unique(X[:, feature]))
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2

            if self.splitter == 'random':
                thresholds = np.random.choice(thresholds, max(1, len(thresholds) // 10), replace=False)

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature], threshold)
                
                # Check sample size in child nodes
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue

                left_counts = np.bincount(y[left_idxs], minlength=self.n_classes)
                right_counts = np.bincount(y[right_idxs], minlength=self.n_classes)
                weighted_impurity = (
                    (len(left_idxs) / total_samples) * self._calculate_impurity(left_counts)
                    + (len(right_idxs) / total_samples) * self._calculate_impurity(right_counts)
                )
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold
                
        # Check information gain
        if best_gain < self.min_impurity_decrease:
            best_feature, best_threshold = None, None
            
        return best_feature, best_threshold

    def _grow(self, X: np.ndarray, y: np.ndarray, depth=0) -> Node:
        counts = np.bincount(y, minlength=self.n_classes)
        
        node = Node(
            value=np.argmax(counts * self.class_weights),  # weighted majority class
            n_samples=counts
        )
        node.impurity = self._calculate_impurity(counts)

        # Stopping conditions
        if (
            counts.sum() == counts.max()  # pure node
            or (self.max_depth is not None and depth >= self.max_depth)
            or counts.sum() < self.min_samples_split
        ):
            return node

        feature, threshold = self._best_split(X, y, node.impurity)
        
        # No good split found
        if feature is None:
            return node
        
        # Recursive growing
        left_idxs, right_idxs = self._split(X[:, feature], threshold)
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[left_idxs], y[left_idxs], depth + 1)
        node.right = self._grow(X[right_idxs], y[right_idxs], depth + 1)

        return node
    
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        
        self.n_features = X.shape[1]
        self.n_classes = len(self.classes_)

        # Compute weights to balance the data out
        class_counts = np.bincount(y_encoded, minlength=self.n_classes)
        total = len(y_encoded)
        self.class_weights = total / (self.n_classes * class_counts)

        self.root = self._grow(X, y_encoded)
        if self.ccp_alpha > 0:
            self._prune_tree()
            
     # ----------------------------
     # Predict
     # ----------------------------
    def _predict_batch(self, X: np.ndarray, node: Node, preds: np.ndarray, idxs: np.ndarray):
        if node.is_leaf():
            preds[idxs] = node.value
            return
        mask = X[idxs, node.feature] <= node.threshold
        if mask.any():
            self._predict_batch(X, node.left, preds, idxs[mask])
        if (~mask).any():
            self._predict_batch(X, node.right, preds, idxs[~mask])

    def predict(self, X):
        X = np.array(X)
        preds = np.empty(X.shape[0], dtype=int)
        self._predict_batch(X, self.root, preds, np.arange(X.shape[0]))
        return self.classes_[preds]  

    # ----------------------------
    # Pruning
    # ----------------------------
    def _subtree_info(self, node):
        if node.is_leaf():
            # Weighted error if this node is leaf
            weighted_counts = node.n_samples * self.class_weights
            total = weighted_counts.sum()
            proportions = weighted_counts / (total if total > 0 else 1)
            node_error = (1 - proportions[node.value]) * total
            return node_error, 1

        R_left, leaves_left = self._subtree_info(node.left)
        R_right, leaves_right = self._subtree_info(node.right)

        R_subtree = R_left + R_right
        n_leaves = leaves_left + leaves_right

        # Cost if we prune this node (turn into leaf)
        weighted_counts = node.n_samples * self.class_weights
        total = weighted_counts.sum()
        proportions = weighted_counts / (total if total > 0 else 1)
        R_node = (1 - proportions[node.value]) * total

        # Effective alpha for pruning this node
        if n_leaves > 1:
            node.alpha = (R_node - R_subtree) / (n_leaves - 1)
        else:
            node.alpha = np.inf  

        return R_subtree, n_leaves

    def _prune_recursive(self, node, alpha_threshold: float):
        if node.is_leaf():
            return node

        # Recurse
        node.left = self._prune_recursive(node.left, alpha_threshold)
        node.right = self._prune_recursive(node.right, alpha_threshold)

        # If this node qualifies for pruning, convert to leaf
        if getattr(node, "alpha", np.inf) <= alpha_threshold:
            node.left = None
            node.right = None
        return node

    def _prune_tree(self):    
        self._subtree_info(self.root)
        self.root = self._prune_recursive(self.root, self.ccp_alpha)

# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Train our implementation ---
tree = DecisionTree(max_depth=5, random_state=42, criterion='gini', ccp_alpha=0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
acc_custom = accuracy_score(y_test, y_pred)

# --- Train sklearn DecisionTree for comparison ---
sk_tree = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='gini', class_weight='balanced')
sk_tree.fit(X_train, y_train)
y_pred_sk = sk_tree.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sk)

print("Custom DecisionTree accuracy:", acc_custom)
print("Sklearn DecisionTree accuracy:", acc_sklearn)

