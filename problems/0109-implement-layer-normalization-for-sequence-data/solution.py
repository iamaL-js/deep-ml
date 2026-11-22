import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
    # Your code here
    X_mean = np.mean(X, axis = -1, keepdims = True)
    X_var = np.var(X, axis = -1, keepdims = True)

    X_hat = (X - X_mean) / np.sqrt(X_var + epsilon)

    y = gamma * X_hat + beta

    return y