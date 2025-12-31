import numpy as np

def init_random(dims, seed=None, scale=0.1):
	L = len(dims) - 1
	# Faux index to maintain notation
	weights = [None]

	rng = np.random.default_rng(seed)

	for l in range(1, L + 1):
		weights.append(rng.standard_normal((dims[l], dims[l-1] + 1)) * scale)

	return weights

def init_constant(dims, value=0.15):
	assert len(dims) >=2
	L = len(dims) - 1
	# Each Index holds the weights going into layer l. Size is L-1
	weights = [None]

	for l in range(1, L + 1):
		weights.append(np.full((dims[l], dims[l-1] + 1), value))

	# Weights at index i refers to the weights going into layer i
	return weights
