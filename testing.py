import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', degree=3, gamma='scale'):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def _polynomial_kernel(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.degree
    
    def _rbf_kernel(self, x1, x2):
        gamma = self.gamma
        if gamma == 'scale':
            gamma = 1.0 / (x1.shape[0] * np.var(x1)) if x1.ndim > 1 else 1.0
        elif gamma == 'auto':
            gamma = 1.0 / x1.shape[0] if x1.ndim > 1 else 1.0
        return np.exp(-gamma * np.sum((x1 - x2) ** 2))
    
    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        
        n_samples1 = X1.shape[0]
        n_samples2 = X2.shape[0]
        K = np.zeros((n_samples1, n_samples2))
        
        for i in range(n_samples1):
            for j in range(n_samples2):
                if self.kernel == 'linear':
                    K[i, j] = self._linear_kernel(X1[i], X2[j])
                elif self.kernel == 'poly':
                    K[i, j] = self._polynomial_kernel(X1[i], X2[j])
                elif self.kernel == 'rbf':
                    K[i, j] = self._rbf_kernel(X1[i], X2[j])
                else:
                    raise ValueError(f"Kernel {self.kernel} not recognized")
        
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize alphas (Lagrange multipliers)
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Simplified SMO-like algorithm
        for _ in range(self.n_iters):
            for idx in range(n_samples):
                # Calculate error
                output = 0
                for i in range(n_samples):
                    output += self.alphas[i] * y_[i] * K[i, idx]
                output += self.b
                error = output - y_[idx]
                
                # Check KKT conditions
                if (y_[idx] * error < -0.001 and self.alphas[idx] < self.lambda_param) or \
                   (y_[idx] * error > 0.001 and self.alphas[idx] > 0):
                    
                    # Update alpha
                    old_alpha = self.alphas[idx]
                    self.alphas[idx] = np.clip(
                        self.alphas[idx] - y_[idx] * error / (K[idx, idx] + 1e-8), 
                        0, 
                        self.lambda_param
                    )
                    
                    # Update bias term
                    self.b = self.b + y_[idx] * (self.alphas[idx] - old_alpha) * K[idx, idx]
        
        # Store support vectors
        support_vector_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y_[support_vector_indices]
        self.support_vector_alphas = self.alphas[support_vector_indices]
        
        # For linear kernel, we can still compute w explicitly
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i in range(len(support_vector_indices)):
                idx = support_vector_indices[i]
                self.w += self.alphas[idx] * y_[idx] * X[idx]

    def predict(self, X):
        if self.kernel == 'linear' and self.w is not None:
            return np.sign(np.dot(X, self.w) - self.b)
        else:
            y_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, sv_y, sv in zip(self.support_vector_alphas, self.support_vector_labels, self.support_vectors):
                    if self.kernel == 'linear':
                        s += alpha * sv_y * self._linear_kernel(X[i], sv)
                    elif self.kernel == 'poly':
                        s += alpha * sv_y * self._polynomial_kernel(X[i], sv)
                    elif self.kernel == 'rbf':
                        s += alpha * sv_y * self._rbf_kernel(X[i], sv)
                s -= self.b
                y_pred[i] = np.sign(s)
            return y_pred


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Try different dataset types
    # Uncomment one of these options:
    
    # Option 1: Linearly separable blobs (works with linear kernel)
    # X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    
    # Option 2: Moons dataset (needs non-linear kernel)
    X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
    
    # Option 3: Circles dataset (needs non-linear kernel)
    # X, y = datasets.make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Choose a kernel: 'linear', 'poly', or 'rbf'
    kernel_type = 'rbf'  # RBF works well with moons and circles
    
    clf = SVM(learning_rate=0.001, lambda_param=0.1, n_iters=1000, kernel=kernel_type)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print(f"SVM with {kernel_type} kernel classification accuracy:", accuracy(y_test, predictions))

    def visualize_svm_decision_boundary():
        # Create a meshgrid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions for all grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and data points
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k')
        
        # Highlight support vectors if available
        if hasattr(clf, 'support_vectors') and clf.support_vectors is not None:
            plt.scatter(clf.support_vectors[:, 0], clf.support_vectors[:, 1], 
                        s=100, linewidth=1, facecolors='none', edgecolors='k')
        
        plt.title(f'SVM Decision Boundary with {kernel_type} Kernel')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.show()

    visualize_svm_decision_boundary()