#!python

"""
PyViennaCL allows you to access and manipulate submatrices and subvectors using
the usual Pythonic apparatus of slices and ranges of objects. ViennaCL provides
object proxies to allow us to do these sub-manipulations in place.

Here, we give some basic examples.
"""

import pyviennacl as p
import numpy as np

# Create some small, simple Vector and Matrix instances
x = p.Vector(6, 1.0)
a = p.Matrix(6, 6, 1.0)

print("x is %s" % x)
print("a is\n%s" % a)

# Scale the first half of the Vector x
x[0:3] *= 2.0

# Show the new x
print("x is now %s" % x)

# Create a smaller matrix from a submatrix of a
b = a[3:6, 3:6] * 4.0

# Set the upper-left corner of the matrix to 4.0s
a[0:3, 0:3] = b

# Show the new a
print("a is now\n%s" % a)

# Represent an operation on a
b = p.sqrt(a)

# Manipulate submatrices of b
b[0:3, 3:6] += b[0:3, 0:3]
b[3:6, 3:6] += b[0:3, 3:6]

# Show b
print("b is\n%s" % b)

# We can also manipulate slices of matrices and of submatrices
c = b[0:6, 2:6]
c[0:6:2, 0:4:2] = c[0:6:2, 0:4:2] * 10.0

# Show b after the proxy update via c
print("b is now\n%s" % b)

# We can do similarly for vectors
x[0:6:2] = x[3:6] * 10.0

# Show x
print("x is now %s" % x)

