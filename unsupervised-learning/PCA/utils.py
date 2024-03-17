import numpy as np

class PCA:
    
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        
        self.mean = np.mean(X, axis=0)
        
        X = X - self.mean
        
        cov = np.cov(X.T)
        
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        
        idxs = np.argsort(eigenvalues.T)[::-1]
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        
        self.components = eigenvectors[:self.n_components]
    
    def transform(self, X):
        
        X = X - self.mean
        
        return np.dot(X, self.components.T)