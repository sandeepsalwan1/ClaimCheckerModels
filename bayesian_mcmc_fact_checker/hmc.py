import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit
from scipy import stats
from scipy.special import softmax
from jax.scipy import stats as jax_stats

# jax log posterior function
# debugged with gen-AI - fixed jax issues
def log_posterior(weights_flat, alpha_flat, X, y, num_features, num_classes):
    weights = jnp.reshape(weights_flat, (num_features, num_classes))

    lp_weights = jnp.sum(jax_stats.norm.logpdf(weights, 0, 3))
    lp_alpha = jnp.sum(jax_stats.norm.logpdf(alpha_flat, 0, 3))
    raw_vals = jnp.dot(X, weights) + alpha_flat
    probabilities = jax.nn.softmax(raw_vals, axis=1)

    indices = jnp.arange(len(y))
    adjusted_prob = jnp.maximum(probabilities[indices, y], 1e-10)
    likelihood = jnp.sum(jnp.log(adjusted_prob))

    log_post = lp_weights + lp_alpha + likelihood
    return log_post

def get_grads(weights_flat, alpha_flat, X, y, num_features, num_classes):
    weight_size = num_features * num_classes
    if len(weights_flat) != weight_size:
        raise ValueError(f"Expected weights_flat to have size {weight_size}, got {len(weights_flat)}")
    
    # calculate gradients
    def grad_fn_weights(w):
        w_reshaped = jnp.reshape(w, (num_features, num_classes))
        return log_posterior(w_reshaped, alpha_flat, X, y, num_features, num_classes)
    
    def grad_fn_alpha(a):
        weights_reshaped = jnp.reshape(weights_flat, (num_features, num_classes))
        return log_posterior(weights_reshaped, a, X, y, num_features, num_classes)
    
    # get gradients
    w_grad = jax.grad(grad_fn_weights)(weights_flat)
    a_grad = jax.grad(grad_fn_alpha)(alpha_flat)

    return w_grad, a_grad

# def log_posterior_direct(params, X, y, num_features, num_classes):
#     # Split the parameters
#     split_idx = num_features * num_classes
#     weights_flat = params[:split_idx]
#     alpha = params[split_idx:]
    
#     weights = jnp.reshape(weights_flat, (num_features, num_classes))
    
#     # Priors
#     lp_weights = jnp.sum(jax_stats.norm.logpdf(weights, 0, 3))
#     lp_alpha = jnp.sum(jax_stats.norm.logpdf(alpha, 0, 3))
    
#     # Likelihood
#     raw_vals = jnp.dot(X, weights) + alpha
#     probabilities = jax.nn.softmax(raw_vals, axis=1)
    
#     # Get probabilities for true classes
#     indices = jnp.arange(len(y))
#     true_probs = probabilities[indices, y]
#     adjusted_probs = jnp.maximum(true_probs, 1e-10)
#     likelihood = jnp.sum(jnp.log(adjusted_probs))
    
#     return lp_weights + lp_alpha + likelihood

# def get_gradient(params, X, y, num_features, num_classes):
#     return jax.grad(lambda p: log_posterior_direct(p, X, y, num_features, num_classes))(params)

class HamiltonianMonteCarlo:
    def __init__(self, model, num_steps=10, num_samples=1000, step_size=0.001, burn_in=200):
        self.model = model
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.step_size = step_size
        self.burn_in = burn_in

        self.X_jax = None
        self.y_jax = None
    
    # flattening/unflattening
    def flatten(self, weights, alpha):
        return np.concatenate([weights.flatten(), alpha.flatten()])
    
    def unflatten(self, flat_params):
        split = self.model.num_classes * self.model.num_features
        weights = flat_params[:split].reshape((self.model.num_features, self.model.num_classes))
        alpha = flat_params[split:]
        return weights, alpha
    
    # get flattened gradients
    def flat_grads(self, X, y, flat_params):
        weights, alpha = self.unflatten(flat_params)
        gradients = self.model.compute_gradients(X, y, weights, alpha)
        return self.flatten(gradients[0], gradients[1])
    
    def flat_grads_jax(self, X, y, flat_params):
        spl = self.model.num_classes * self.model.num_features
        weights_flat = flat_params[:spl]
        alpha_flat = flat_params[spl:]

        X_jax = self.X_jax if self.X_jax is not None else jnp.array(X)
        y_jax = self.y_jax if self.y_jax is not None else jnp.array(y)
        
        weights_jax_flat = jnp.array(weights_flat)
        alpha_jax_flat = jnp.array(alpha_flat)

        weight_gradients, alpha_gradients = get_grads(weights_jax_flat, alpha_jax_flat, X_jax, y_jax, self.model.num_features, self.model.num_classes)

        return np.concatenate([np.array(weight_gradients).flatten(), np.array(alpha_gradients).flatten()])
    
    def get_log_post_jax(self, X, y, weights, alpha):
        X_jax = self.X_jax if self.X_jax is not None else jnp.array(X)
        y_jax = self.y_jax if self.y_jax is not None else jnp.array(y)

        weights_flat = weights.flatten()
        weights_jax_flat = jnp.array(weights_flat)
        alpha_jax_flat = jnp.array(alpha)

        log_post = log_posterior(weights_jax_flat, alpha_jax_flat, X_jax, y_jax, self.model.num_features, self.model.num_classes)

        return float(log_post)
    
    # hamiltonian monte carlo sampling
    # main sampling function
    def hmc_sampling(self, X, y):
        self.X_jax = jnp.array(X)
        self.y_jax = jnp.array(y)

        weights_initial = self.model.weights
        alpha_initial = self.model.alpha
        params_initial = self.flatten(weights_initial, alpha_initial)
        num_params = len(params_initial)
        num_accepted = 0

        log_post = []
        weight_samples = []
        alpha_samples = []

        print("Starting sampling")

        for i in range(self.num_samples + self.burn_in): # account for burn in value
            momentum_initial = np.random.normal(0, 1, num_params)
            log_posterior_initial = self.get_log_post_jax(X, y, weights_initial, alpha_initial)
            potential_initial = -log_posterior_initial
            kinetic_initial = 0.5 * momentum_initial.T.dot(momentum_initial)
            # H(theta, p) = U(theta) + K(p) (potential + kinetic)
            hamiltonian_intial = potential_initial + kinetic_initial
            log_post.append(log_posterior_initial)
            # print("Set initial hamiltonian")

            # leapfrog
            params_prop = params_initial.copy()
            momentum_prop = momentum_initial.copy()

            flat_gradients = self.flat_grads_jax(X, y, params_prop)
            momentum_prop += 0.5 * self.step_size * flat_gradients

            # print("Starting leapfrog")
            for j in range(self.num_steps):
                # print(f"Leapfrog step {j + 1}/{self.num_steps}")
                params_prop += self.step_size * momentum_prop
                if j < self.num_steps - 1:
                    flat_gradients = self.flat_grads_jax(X, y, params_prop)
                    momentum_prop += self.step_size * flat_gradients
            # print("Finished leapfrog")
                
            flat_gradients = self.flat_grads_jax(X, y, params_prop)
            momentum_prop += 0.5 * self.step_size * flat_gradients
            momentum_prop = -momentum_prop

            # proposted hamiltonian
            weights_prop, alpha_prop = self.unflatten(params_prop)
            # same procedure as intial
            log_posterior_prop = self.get_log_post_jax(X, y, weights_prop, alpha_prop)
            potential_prop = -log_posterior_prop
            kinetic_prop = 0.5 * momentum_prop.T.dot(momentum_prop)
            hamiltonian_prop = potential_prop + kinetic_prop
            # print("Set proposed hamiltonian")

            # metropolis acceptance

            # find probability of acceptance
            delta_h = hamiltonian_prop - hamiltonian_intial
            prob_acceptance = min(np.exp(-delta_h), 1)
            print(f"Delta H: {delta_h}, Probability of acceptance: {prob_acceptance}")

            if np.random.uniform(0, 1) < prob_acceptance:
                # accept sample and update values accordingly
                params_initial = params_prop
                weights_initial, alpha_initial = weights_prop, alpha_prop

                if i >= self.burn_in:
                    num_accepted += 1
                    weight_samples.append(weights_initial.copy())
                    alpha_samples.append(alpha_initial.copy())
            elif i >= self.burn_in:
                weight_samples.append(weights_initial.copy())
                alpha_samples.append(alpha_initial.copy())
            
            # if (i + 1) % 100 == 0:
            print(f"Sample {i + 1}/{self.num_samples + self.burn_in} completed")

        acceptance_rate = 0
        if self.num_samples != 0:
            acceptance_rate = num_accepted / self.num_samples
        
        print(f"Finished sampling, acceptance rate = {acceptance_rate}")

        self.X_jax = None
        self.y_jax = None

        return np.array(weight_samples), np.array(alpha_samples), log_post, acceptance_rate