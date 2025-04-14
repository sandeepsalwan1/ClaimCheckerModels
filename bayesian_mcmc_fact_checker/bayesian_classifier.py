import numpy as np
import scipy.stats as stats
from scipy.special import softmax
import jax
import jax.numpy as jnp
from jax import grad, jit


# bayesian news classifier using markov chain monte carlo
class BayesianClassifier:
    def __init__(self, num_classes = 6, num_features = 768):
        self.num_classes = num_classes
        self.num_features = num_features

        # used gen-AI to debug - realized I was using a suboptimal std and also incorrectly formatting the matrix
        self.weights = np.random.normal(0, 0.1, (self.num_features, self.num_classes))
        self.alpha = np.random.normal(0, 0.1, self.num_classes)

        self.weight_samp = []
        self.alpha_samp = []

    def log_prior(self, weights, alpha):
        # assuming mean of 0 and std of 3
        lp_weights = np.sum(stats.norm.logpdf(weights, 0, 3))
        lp_alpha = np.sum(stats.norm.logpdf(alpha, 0, 3))
        return lp_weights + lp_alpha
    
    def log_likelihood(self, X, y, weights, alpha):
        # debug purposes, remove print statements later
        # print(f'X shape: {X.shape}')
        # print(f'weights shape: {weights.shape}')
        # print(f'alpha shape: {alpha.shape}')
        # calculate raw values
        raw_vals = np.dot(X, weights) + alpha
        probabilities = softmax(raw_vals, axis=1)

        # calculate log likelihood
        likelihood = 0
        for i in range(len(y)):
            adjusted_prob = max(probabilities[i, y[i]], 1e-10) # adjusting the probability to avoid log of 0
            likelihood += np.log(adjusted_prob)
        return likelihood
    
    # manual
    def log_posterior_prob(self, X, y, weights, alpha):
        log_post = self.log_prior(weights, alpha) + self.log_likelihood(X, y, weights, alpha)
        return log_post
    
    # manual computation
    def compute_gradients(self, X, y, weights, alpha):
        gradient_weights = np.zeros_like(weights)
        gradient_alpha = np.zeros_like(alpha)
        epsilon = 1e-6

        orig_log_posterior = self.log_posterior_prob(X, y, weights, alpha)

        num_features = weights.shape[0]
        num_classes = weights.shape[1]

        # weight gradients
        for i in range(num_features):
            for j in range(num_classes):
                # formula: f(x+epsilon) - f(x-epsilon) / 2*epsilon
                weights_upper = weights.copy()
                weights_lower = weights.copy()
                weights_upper[i, j] += epsilon
                weights_lower[i, j] -= epsilon
                log_posterior_upper = self.log_posterior_prob(X, y, weights_upper, alpha)
                log_posterior_lower = self.log_posterior_prob(X, y, weights_lower, alpha)
                gradient_weights[i, j] = (log_posterior_upper - log_posterior_lower) / (2*epsilon)
                # gradient_weights[i, j] = np.clip(gradient_weights[i, j], -100, 100) # clip to avoid overflow
        
        # alpha gradients
        # used gen-AI to debug - realized that I was mixing up dimensions and having a ton of index errors due to that
        for i in range(num_classes):
            # formula: f(x+epsilon) - f(x-epsilon) / 2*epsilon, except with alpha this time
            alpha_upper = alpha.copy()
            alpha_lower = alpha.copy()
            alpha_upper[i] += epsilon
            alpha_lower[i] -= epsilon
            log_posterior_upper = self.log_posterior_prob(X, y, weights, alpha_upper)
            log_posterior_lower = self.log_posterior_prob(X, y, weights, alpha_lower)
            gradient_alpha[i] = (log_posterior_upper - log_posterior_lower) / (2*epsilon)

        return gradient_weights, gradient_alpha
    
    def predict_probs(self, X):
        raw_vals = np.dot(X, self.weights) + self.alpha
        probabilities = softmax(raw_vals, axis=1)
        return probabilities

    def predict(self, X):
        raw_vals = np.dot(X, self.weights) + self.alpha
        probabilities = softmax(raw_vals, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
