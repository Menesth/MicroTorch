# MicroTorch

Implementation from scratch of Torch's backward method in view of training loop.

Based on A. Karpathy's micrograd.

Note: if N = number of samples and p = number of parameters, then the feature vector X must be of shape (N, p), even if p = 1. The target vector y must be of shape (N, ).
