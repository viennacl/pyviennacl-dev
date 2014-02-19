#!python

"""
Sparse matrix support is still limited in PyViennaCL. The construction
of spare matrices from host data is supported, as is sparse matrix-vector
multiplication and the use of iterative solvers (see iterative-solvers.py
in this directory). However, no other operations have yet been implemented,
and SciPy support is rudimentary.

Here, we demonstrate the construction of a CompressedMatrix instance, and the
calculation of a matrix-vector product.

Other sparse matrix formats are available, such as the CoordinateMatrix,
ELLMatrix and HybridMatrix. They are constructed and used identically to the
CompressedMatrix type.
"""

import pyviennacl as p
import random

# First, we create an empty 5 x 5 CompressedMatrix:
A = p.CompressedMatrix(5, 5)

# Let's set some random values of A.
#
# Importantly, setting individual elements of a PyViennaCL sparse matrix is
# not nearly as expensive as setting individual elements of a dense matrix or
# vector, since in the sparse matrix case, the elements are cached on the host
# and only transferred to the device when they are needed for some computation.
for i in range(6):
    x = random.randrange(0, 4, 1)
    y = random.randrange(0, 4, 1)
    A[x, y] = random.random()

print("A is:\n%s" % A.value)

# Now, let's construct a simple vector of 5 elements.
b = p.Vector(5, 3.142)

print("b is %s" % b)

# Now, represent the product:
c = A * b

# And the result is only computed when we need to print it:
print("A * b = c is %s" % c)

