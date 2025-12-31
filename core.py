import numpy as np

OUTPUT_IDENTITY = 0
OUTPUT_TANH = 1
OUTPUT_SIGN = 2


def compute_gradients(X, Y, weights, out_type, lam: float):
	assert lam >= 0.0
	assert X.ndim == 2

	L = len(weights) - 1
	N = X.shape[0]
	assert Y.shape[0] == N

	act, signals = forward_prop(X, weights, out_type)
	delta = back_prop(Y, weights, act, signals, out_type)

	G = [None] * (L + 1)

	for l in range(1, L + 1):
		G[l] = (delta[l].T @ act[l - 1]) / N

		if lam > 0.0:
			G[l][:, 1:] += 2 * (lam / N) * weights[l][:, 1:]

	return G

def forward_prop(X, weights, out_type):
	assert X.ndim == 2
	L = len(weights) - 1
	N = X.shape[0]
	act = []
	# Faux index to maintain notation
	signals = [None]

	# Add bias
	X_bias = np.c_[np.ones((N,1), dtype=X.dtype), X]
	act.append(X_bias)

	# For each layer, compute signal and apply tanh
	for l in range(1, L):
		S =  X_bias @ weights[l].T
		signals.append(S)
		X_bias = np.tanh(S)
		X_bias = np.c_[np.ones((N, 1), dtype=X.dtype), X_bias]
		act.append(X_bias)

	# Compute final layer w/ desired output type
	S = X_bias @ weights[L].T
	signals.append(S)
	if out_type == OUTPUT_IDENTITY:
		act.append(S)
	elif out_type == OUTPUT_TANH:
		act.append(np.tanh(S))
	elif out_type == OUTPUT_SIGN:
		act.append(np.sign(S))
	else:
		raise ValueError("Invalid output type")

	return act, signals

def back_prop(Y, weights, act, signals, out_type):
	if out_type == OUTPUT_SIGN:
		raise ValueError("Sign output is for inference only, derivative is undefined for the sign function")
	L = len(weights) - 1
	N = act[0].shape[0]
	Y = Y.reshape(N,1)

	delta = [0] * (L + 1)

	if out_type == OUTPUT_IDENTITY:
		delta[L] = 0.5 * (act[L] - Y)
	elif out_type == OUTPUT_TANH:
		delta[L] = 0.5 * (act[L] - Y) * (1 - act[L] ** 2)
	else:
		raise ValueError("Invalid output type")

	for l in reversed(range(1, L)):
		#Strip the bias off of the act, much faster than computing tanh from the signals
		a_nbias = act[l][:, 1:]
		theta_prime = 1 - a_nbias ** 2
		w_nbias = weights[l + 1][:, 1:]
		delta[l] = (delta[l + 1] @ w_nbias) * theta_prime

	return delta	