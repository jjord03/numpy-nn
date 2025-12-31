import numpy as np
from core import forward_prop

def compute_error(X, Y, weights, out_type, lam: float = 0.0):
	L = len(weights) - 1
	N = X.shape[0]
	Y = Y.reshape(N,1)

	activations, signals = forward_prop(X, weights, out_type)
	err = activations[L] - Y
	E_in = (err * err).sum() / (4 * N)

	if lam > 0:
		decay = 0
		for l in range(1, L + 1):
			decay += (weights[l][:, 1:] ** 2).sum()
		E_in += (lam / N) * decay

	return E_in