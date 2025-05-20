from __future__ import print_function
import numpy as np
from cvxopt import matrix, solvers

np.random.seed(22)

# Generate synthetic data
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -np.ones(N)), axis=0) # labels

# Create the matrix for quadratic programming
Xbar = np.concatenate((X0, -X1), axis=0)
Q = matrix(Xbar.dot(Xbar.T))
p = matrix(np.ones((2*N, 1)))  # objective function 1/2 lambda^T*Q*lambda - 1^T*lambda

# Set up constraints
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros((1, 1)))

solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)

if sol['status'] != 'optimal':
    raise ValueError("Optimal solution not found")

lambda_vals = np.array(sol['x'])

# compute w and b
w = Xbar.T.dot(lambda_vals)
support_vectors = np.where(lambda_vals > 1e-11)[0]

if len(support_vectors) == 0:
    raise ValueError("No support vectors found")

b = np.mean(y[support_vectors].reshape(-1, 1) - X[support_vectors,:].dot(w))


print("Number of support vectors:", len(support_vectors))
print("w:", w.T)
print("b:", b)


























