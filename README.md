# Minimal NumPy Neural Net (From Scratch)

This repo is a small neural network implementation written in plain NumPy.

I built this to learn what a neural network is doing ''under the hood'' so I can understand it deeply and apply the same ideas more confidently in real projects. The focus is on the fundamentals: forward propagation, backpropagation, and training loops.

I implemented the math directly (including backpropagation and a numeric gradient check) to verify correctness and to make the training behavior easier to reason about.

## What’s inside

- `core.py` — forward propagation + backpropagation (analytic gradients)
- `init.py` — weight initialization helpers
- `losses.py` — MSE loss (optionally with L2 weight decay)
- `optim.py` — optimizers (SGD and variable learning-rate GD)
- `gradcheck.py` — numeric gradient checker (finite differences)
- `run_gradcheck.py` — small script that runs the gradient check on a toy example

