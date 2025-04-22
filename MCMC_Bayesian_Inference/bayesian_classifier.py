import numpy as np
import scipy.stats as stats
from scipy.special import softmax
import jax
import jax.numpy as jnp
from jax import grad, jit


# bayesian news classifier model
class BayesianClassifier:
    def __init__(self, num_classes = 6, num_features = 768):
        self.num_classes = num_classes
        self.num_features = num_features

        self.weights = np.random.normal(0, 0.1, (self.num_features, self.num_classes))
        self.alpha = np.random.normal(0, 0.1, self.num_classes)

        self.weight_samp = []
        self.alpha_samp = []
    
    def predict_probs(self, X):
        raw_vals = np.dot(X, self.weights) + self.alpha
        probabilities = softmax(raw_vals, axis=1)
        return probabilities

    def predict(self, X):
        raw_vals = np.dot(X, self.weights) + self.alpha
        probabilities = softmax(raw_vals, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
