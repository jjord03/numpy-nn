import numpy as np
from init import init_constant
from core import OUTPUT_IDENTITY, OUTPUT_TANH
from gradcheck import gradcheck

x1 = np.array([[5, 6]])
y = np.array([1])

d = [2, 2, 1]

weights = init_constant(d, value=0.15)

print("IDENTITY:")
gradcheck(x1, y, weights, OUTPUT_IDENTITY, eps=1e-5)

print("\nTANH")
gradcheck(x1, y, weights, OUTPUT_TANH, eps=1e-5)

