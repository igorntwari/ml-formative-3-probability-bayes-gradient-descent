"""Gradient descent for the model  y_hat = X @ m + b  (Formative 3, Part 4).

Everything is written with matrices/vectors, not scalars.
    X : (n, 2) data matrix      m : (2,) weights      b : (n,) bias vector
    J(m, b) = (1/n) * sum((y_hat - y)^2)

Chain rule, with e = y_hat - y:
    dJ/de = (2/n) * e
    de/dm = X        ->  dJ/dm = (2/n) * X^T @ e
    de/db = I        ->  dJ/db = (2/n) * e
"""
import numpy as np

try:  # SciPy >= 1.15: accurate derivative routine
    from scipy.differentiate import derivative as _scipy_derivative

    def _derivative(f, x):
        return float(_scipy_derivative(f, x).df)
except ImportError:  # older SciPy (Python 3.9 tops out at SciPy 1.13)
    from scipy.optimize import approx_fprime

    def _derivative(f, x):
        return float(approx_fprime(np.array([x], dtype=float),
                                   lambda v: float(f(v[0])), 1e-6)[0])


# ---------------------------------------------------------------- problem data
X = np.array([[1.0, 3.0], [4.0, 10.0]])
Y = np.array([5.0, 6.0])
M0 = np.array([-1.0, 2.0])
B0 = np.array([1.0, 1.0])


# ---------------------------------------------------------------- model + cost
def predict(X, m, b):
    """y_hat = X @ m + b"""
    return X @ m + b


def mse_cost(X, y, m, b):
    """J = (1/n) * e^T e, where e = y_hat - y"""
    error = predict(X, m, b) - y
    return float(error @ error) / len(y)


# ---------------------------------------------------------------- SciPy derivative
def derivative_of(equation, x):
    """Derivative of a one-variable equation f(x) at the point x, using SciPy."""
    return _derivative(np.vectorize(equation, otypes=[float]), x)


def _with_value(params, index, value):
    """Copy of params with one entry replaced."""
    changed = np.array(params, dtype=float)
    changed[index] = value
    return changed


def scipy_gradient(X, y, m, b):
    """Gradient of J from SciPy: one partial derivative per parameter."""
    params = np.concatenate([m, b])
    n_m = len(m)

    def cost_of(p):
        return mse_cost(X, y, p[:n_m], p[n_m:])

    grads = np.array([
        derivative_of(lambda t, i=i: cost_of(_with_value(params, i, t)), params[i])
        for i in range(len(params))
    ])
    return grads[:n_m], grads[n_m:]


# ---------------------------------------------------------------- analytic gradient
def analytic_gradient(X, y, m, b):
    """Gradient of J from the chain rule, in matrix form."""
    n = len(y)
    error = predict(X, m, b) - y
    grad_m = (2 / n) * (X.T @ error)
    grad_b = (2 / n) * error
    return grad_m, grad_b


# ---------------------------------------------------------------- one update
def gradient_step(X, y, m, b, lr, grad_fn=analytic_gradient):
    """One update. Returns every intermediate value so each step stays visible."""
    y_hat = predict(X, m, b)
    error = y_hat - y
    grad_m, grad_b = grad_fn(X, y, m, b)
    return {
        "m": m, "b": b,
        "y_hat": y_hat, "error": error,
        "cost": mse_cost(X, y, m, b),
        "grad_m": grad_m, "grad_b": grad_b,
        "new_m": m - lr * grad_m,
        "new_b": b - lr * grad_b,
    }


def run_gradient_descent(X, y, m, b, lr, n_iterations, grad_fn=analytic_gradient):
    """Repeat gradient_step. Returns (steps, final_m, final_b)."""
    steps = []
    for _ in range(n_iterations):
        step = gradient_step(X, y, m, b, lr, grad_fn)
        steps.append(step)
        m, b = step["new_m"], step["new_b"]
    return steps, m, b


# ---------------------------------------------------------------- reporting
def build_history(steps, X, y, final_m, final_b):
    """Per-iteration series (iteration 0 = start, last = after final update)."""
    ms = [s["m"] for s in steps] + [final_m]
    bs = [s["b"] for s in steps] + [final_b]
    costs = [s["cost"] for s in steps] + [mse_cost(X, y, final_m, final_b)]
    return {"m": np.array(ms), "b": np.array(bs), "cost": np.array(costs)}


def describe_step(number, step, lr):
    """Print one update with all intermediate results."""
    print(f"--- Update {number} ---")
    print("m      =", step["m"], "  b =", step["b"])
    print("y_hat  = X @ m + b =", step["y_hat"])
    print("error  = y_hat - y =", step["error"])
    print(f"cost J = {step['cost']:.6f}")
    print("dJ/dm  = (2/n) X^T e =", step["grad_m"])
    print("dJ/db  = (2/n) e     =", step["grad_b"])
    print(f"new m  = m - {lr} * dJ/dm =", step["new_m"])
    print(f"new b  = b - {lr} * dJ/db =", step["new_b"])
    print()