"""Implementations of the regression methods used in Project 1."""

import numpy as np


def _mse_loss(y, tx, w):
	"""Return the mean squared error with the lecture's 0.5 factor."""
	residual = y - tx @ w
	return 0.5 * np.mean(residual ** 2)


def _sigmoid(scores):
	"""Compute sigmoid values without overflowing for large scores."""
	values = np.empty_like(scores, dtype=float)
	positive = scores >= 0
	values[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
	exp_scores = np.exp(scores[~positive])
	values[~positive] = exp_scores / (1.0 + exp_scores)
	return values


def _logistic_loss(y, tx, w):
	"""Return the unregularized logistic loss."""
	scores = tx @ w
	# log(1 + exp(score)) - y * score, evaluated stably.
	return np.sum(np.logaddexp(0.0, scores) - y * scores)


def _check_inputs(y, tx, w):
	"""Validate the vector and design-matrix shapes shared by the methods."""
	if y.ndim != 1 or w.ndim != 1 or tx.ndim != 2:
		raise ValueError("y and w must be 1D arrays and tx must be 2D")
	if tx.shape[0] != y.shape[0] or tx.shape[1] != w.shape[0]:
		raise ValueError("incompatible shapes for y, tx, and w")


def mean_squared_error_gd(y, tx, initialw, maxiters, gamma):
	"""Linear regression with full-batch gradient descent."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	w = np.array(initialw, dtype=float, copy=True)
	_check_inputs(y, tx, w)

	sample_count = y.shape[0]
	for _ in range(maxiters):
		residual = y - tx @ w
		gradient = -(tx.T @ residual) / sample_count
		w -= gamma * gradient

	return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initialw, maxiters, gamma):
	"""Linear regression with stochastic gradient descent (batch size one)."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	w = np.array(initialw, dtype=float, copy=True)
	_check_inputs(y, tx, w)

	sample_count = y.shape[0]
	for _ in range(maxiters):
		index = np.random.randint(sample_count)
		residual = y[index] - tx[index] @ w
		gradient = -tx[index] * residual
		w -= gamma * gradient

	return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
	"""Linear regression solved with the normal equations."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	w = np.linalg.solve(tx.T @ tx, tx.T @ y)
	return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
	"""Ridge regression solved with regularized normal equations."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	feature_count = tx.shape[1]
	sample_count = tx.shape[0]
	system = tx.T @ tx + 2.0 * sample_count * lambda_ * np.eye(feature_count)
	w = np.linalg.solve(system, tx.T @ y)
	# The reported loss intentionally excludes the ridge penalty.
	return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initialw, maxiters, gamma):
	"""Binary logistic regression with gradient descent."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	w = np.array(initialw, dtype=float, copy=True)
	_check_inputs(y, tx, w)

	for _ in range(maxiters):
		probabilities = _sigmoid(tx @ w)
		gradient = tx.T @ (probabilities - y)
		w -= gamma * gradient

	return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initialw, maxiters, gamma):
	"""Regularized binary logistic regression with gradient descent."""
	y = np.asarray(y, dtype=float)
	tx = np.asarray(tx, dtype=float)
	w = np.array(initialw, dtype=float, copy=True)
	_check_inputs(y, tx, w)

	for _ in range(maxiters):
		probabilities = _sigmoid(tx @ w)
		gradient = tx.T @ (probabilities - y) + 2.0 * lambda_ * w
		w -= gamma * gradient

	# The reported loss intentionally excludes the regularization penalty.
	return w, _logistic_loss(y, tx, w)
