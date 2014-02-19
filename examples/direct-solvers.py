#!python

"""
PyViennaCL provides direct solvers for dense triangular linear systems.
The API is documented in ``help(pyviennacl.linalg.solve)``. In particular,
the solver to use is determined by the tag instance supplied to the ``solve``
function.

LU factorisation is planned for a later release.

Here, we demonstrate the use of the direct triangular solver for an
upper triangular system; other forms and solvers are supported by the choice
of a different solver tag class.
"""

import pyviennacl as p
import numpy as np
import random

# We want a square N x N system.
N = 5 

# Create a NumPy matrix with float32 precision to hold the data on the host.
# Firstly, we create an empty matrix, then fill the upper triangle with values.
A = np.zeros((N, N), dtype = np.float32)

for i in range(N):
    for j in range(N):
        if j >= i:
            A[i, j] = np.float32(random.randint(0,1000) / 100.0)

# Transfer the system matrix to the compute device
A = p.Matrix(A)

print("A is\n%s" % A)

# Create a right-hand-side vector on the host with random elements
# and transfer it to the compute device
b = p.Vector(np.random.rand(N).astype(np.float32))

print("b is %s" % b)

# Solve the system; note the choice of tag to denote an upper triangular system
x = p.solve(A, b, p.upper_tag())

# Copy the solution from the device to host and display it
print("Solution of Ax = b for x:\n%s" % x)

