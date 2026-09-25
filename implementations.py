import numpy as np


def compute_mse(y, tx, w):
    """Compute the mean squared error loss.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        w: shape=(D,). Model parameters.

    Returns:
        Scalar mean squared error with the 0.5 factor used in the course.
    """
    err = y - tx.dot(w)
    return 0.5 * np.mean(err**2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Train a linear regression model using  gradient descent.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        initial_w: shape=(D,). Initial model parameters.
        max_iters: Number of gradient descent iterations.
        gamma: Gradient descent learning rate.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar mean squared error at the final parameters.
    """
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(y)
        w -= gamma * grad

    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Train a linear regression model using stochastic gradient descent.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        initial_w: shape=(D,). Initial model parameters.
        max_iters: Number of stochastic gradient descent iterations.
        gamma: Gradient descent learning rate.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar mean squared error at the final parameters.
    """
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        index = np.random.randint(len(y))

        err = y[index] - tx[index].dot(w)
        grad = -tx[index] * err

        w -= gamma * grad

    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Compute the least-squares solution using the normal equations.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.

    Returns:
        w: shape=(D,). Least-squares model parameters.
        loss: Scalar mean squared error at the solution.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Compute ridge regression using the normal equations.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        lambda_: Regularization parameter.

    Returns:
        w: shape=(D,). Ridge-regression model parameters.
        loss: Scalar mean squared error.
    """
    n = tx.shape[0]
    d = tx.shape[1]

    a = tx.T.dot(tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)

    return w, compute_mse(y, tx, w)


def sigmoid(t):
    """Compute the sigmoid function in a numerically stable way.

    Args:
        t: Array-like input values.

    Returns:
        Sigmoid values with the same shape as t.
    """
    t = np.asarray(t, dtype=float)
    result = np.empty_like(t)

    positive = t >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-t[positive]))

    exp_t = np.exp(t[~positive])
    result[~positive] = exp_t / (1.0 + exp_t)

    return result


def compute_logistic_loss(y, tx, w):
    """Compute the average binary logistic loss.

    Args:
        y: shape=(N,). Binary labels in {0, 1}.
        tx: shape=(N, D). Input data matrix.
        w: shape=(D,). Model parameters.

    Returns:
        Scalar average logistic loss.
    """
    scores = tx.dot(w)
    return np.mean(np.logaddexp(0.0, scores) - y * scores)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Train binary logistic regression using gradient descent.

    Args:
        y: shape=(N,). Binary labels in {0, 1}.
        tx: shape=(N, D). Input data matrix.
        initial_w: shape=(D,). Initial model parameters.
        max_iters: Number of gradient-descent iterations.
        gamma: Gradient-descent step size.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar logistic loss at the final parameters.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n = len(y)

    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) / n
        w -= gamma * grad

    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Train regularized binary logistic regression using gradient descent.

    Args:
        y: shape=(N,). Binary labels in {0, 1}.
        tx: shape=(N, D). Input data matrix.
        lambda_: Non-negative regularization parameter.
        initial_w: shape=(D,). Initial model parameters.
        max_iters: Number of gradient-descent iterations.
        gamma: Gradient-descent step size.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar unregularized logistic loss.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n = len(y)

    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) / n
        grad += 2 * lambda_ * w
        w -= gamma * grad

    return w, compute_logistic_loss(y, tx, w)
