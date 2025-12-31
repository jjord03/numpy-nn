import numpy as np
from core import compute_gradients
from losses import compute_error

def num_gradient(X, Y, weights, out_type, eps=1e-5):
	L = len(weights) - 1
	num_G = [None] + [np.zeros_like(weights[l]) for l in range(1, len(weights))]

	for l in range(1, L + 1):
		for i in range(weights[l].shape[0]):
			for j in range(weights[l].shape[1]):

				weights[l][i,j] += eps
				err_pos = compute_error(X, Y, weights, out_type, 0.0)

				weights[l][i,j] -= 2 * eps
				err_neg = compute_error(X, Y, weights, out_type, 0.0)

				weights[l][i,j] += eps

				num_G[l][i, j] = (err_pos - err_neg) / (2 * eps)


	return num_G

def gradcheck(X, Y, weights, out_type, eps=1e-5, atol=1e-7, rtol=1e-4):
	G = compute_gradients(X, Y, weights, out_type, 0.0)

	num_G = num_gradient(X, Y, weights, out_type, eps=eps)

	max_rel_error = 0.0
	worst = None

	for l in range(1, len(weights)):
		denom = np.abs(G[l]) + np.abs(num_G[l]) + 1e-12
		rel = np.abs(G[l] - num_G[l]) / denom
		idx = np.unravel_index(np.argmax(rel), rel.shape)
		layer_max = float(rel[idx])

		if layer_max > max_rel_error:
			max_rel_error = layer_max
			worst = (l, idx, float(G[l][idx]), float(num_G[l][idx]), layer_max)


	print("max_rel_error", max_rel_error)
	print("worst:", worst)