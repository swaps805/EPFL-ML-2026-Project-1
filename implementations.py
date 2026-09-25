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


def compute_stoch_gradient(y, tx, w):
    """Compute the gradient of the mean squared error on a mini-batch.

    Args:
        y: shape=(N,). Target values for the mini-batch.
        tx: shape=(N, D). Input data for the mini-batch.
        w: shape=(D,). Model parameters.

    Returns:
        grad: shape=(D,). Gradient of the loss with respect to w.
        err: shape=(N,). Prediction errors y - tx @ w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(y)
    return grad, err


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Train linear regression using batch gradient descent.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        initial_w: shape=(D,). Initial model parameters.
        max_iters: Number of gradient-descent iterations.
        gamma: Gradient-descent step size.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar mean squared error at the final parameters.
    """
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        grad, _ = compute_stoch_gradient(y, tx, w)
        w -= gamma * grad

    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Train linear regression using mini-batch stochastic gradient descent.

    A mini-batch is sampled using batch_iter at every iteration. The
    gradient is computed on the sampled mini-batch, while the reported
    loss is computed on the full training set.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        initial_w: shape=(D,). Initial model parameters.
        batch_size: Number of examples in each mini-batch.
        max_iters: Number of SGD iterations.
        gamma: SGD step size.

    Returns:
        w: shape=(D,). Final model parameters.
        loss: Scalar mean squared error on the full dataset.
    """
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
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

    The regularization term is (lambda_/2) * ||w||^2.

    The objective is:
        0.5/N * ||y - Xw||^2 + lambda_/2 * ||w||^2

    which gives the normal equations:
        (X^T X / N + lambda_ I) w = X^T y / N

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        lambda_: Non-negative regularization parameter.

    Returns:
        w: shape=(D,). Ridge-regression model parameters.
        loss: Scalar unregularized mean squared error.
    """
    n = tx.shape[0]
    d = tx.shape[1]

    a = tx.T.dot(tx) / n + lambda_ * np.eye(d)
    b = tx.T.dot(y) / n

    w = np.linalg.solve(a, b)

    # The project requires the returned loss to exclude the penalty term.
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

    The regularization term is (lambda_/2) * ||w||^2, so its gradient
    is lambda_ * w. The returned loss is the unregularized logistic loss.

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
        grad += lambda_ * w
        w -= gamma * grad

    return w, compute_logistic_loss(y, tx, w)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate mini-batches from a dataset.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data matrix.
        batch_size: Number of examples in each mini-batch.
        num_batches: Number of mini-batches to generate.
        shuffle: Whether to randomly choose mini-batch starting positions.

    Yields:
        Tuples (y_batch, tx_batch) containing matching mini-batches.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)
    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start
        end_index = start_index + batch_size
        yield y[start_index:end_index], tx[start_index:end_index]
