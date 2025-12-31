import numpy as np
from core import compute_gradients
from losses import compute_error


def SGD(D, weights, out_type, max_iterations, eta=0.01, lam: float = 0.0):
	Y = D[:, 0]
	X = D[:, 1:]
	N = X.shape[0]

	for _ in range(max_iterations):
		i = np.random.randint(0, N)

		x = X[i:i+1]
		y = Y[i:i+1]

		G = compute_gradients(x, y, weights, out_type, lam)

		for l in range(1, len(weights)):
			weights[l] -= eta * G[l]

	return weights


def VLRGD(D, weights, out_type, max_iterations, eta=0.01, lam: float = 0.0):
	Y = D[:, 0]
	X = D[:, 1:]

	E_last, G_last = compute_error(X, Y, weights, out_type, lam), None
	last_accepted = True

	for _ in range(max_iterations):

		if last_accepted:
			G_last = compute_gradients(X, Y, weights, out_type, lam)
		
		w_new = [None] * len(weights)
		for l in range(1, len(weights)):
			w_new[l] = weights[l] - eta * G_last[l]

		E_new = compute_error(X, Y, w_new, out_type, lam)

		if E_new < E_last:
			weights = w_new
			E_last = E_new
			eta *= 1.05
			last_accepted = True
		else:
			eta *= 0.65
			last_accepted = False

	return weights