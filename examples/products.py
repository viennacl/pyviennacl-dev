#!python

"""
The overloading of the * operator in PyViennaCL is not directly compatible with
the usage in NumPy. In PyViennaCL, given the emphasis on linear algebra, we have
attempted to make the usage of * as natural as possible.

The semantics are as follows:
* Scalar * scalar -> scalar;
* scalar * vector -> scaled vector;
* scalar * matrix -> scaled matrix;
* vector * vector -> undefined;
* vector * matrix -> undefined;
* matrix * vector -> matrix-vector product;
* matrix * matrix -> matrix-matrix product.

Of course, there exist other products in some cases, which is why the * operator
is sometimes undefined. For instance, in the case of `vector * vector`, we
could have either a dot product, an outer product, a cross product, or an
elementwise product. The cross product is not currently implemented in
PyViennaCL, but this still leaves three cases from which to choose.

Here, we demonstrate the different notation for these products.
"""

import pyviennacl as p
import numpy as np

# Let's construct some random 1-D and 2-D arrays
v = np.random.rand(5)
w = np.random.rand(5)

f = np.random.rand(5, 5)
g = np.random.rand(5, 5)

# Now transfer them to the compute device
x, y = p.Vector(v), p.Vector(w)
a, b = p.Matrix(f), p.Matrix(g)

print("a is\n%s" % a)
print("b is\n%s" % b)

print("x is %s" % x)
print("y is %s" % y)

#
# Scaling
#

# Represent the scaling of x by 2.0
z = x * 2.0

# Compute and print the result
print("x * 2.0 = %s" % z)

# Represent the scaling of a by 2.0
c = a * 2.0

# Compute and print the result
print("a * 2.0 =\n%s" % c)

#
# Vector products
#

# Represent the dot product of x and y
d = x.dot(y) # or p.dot(x, y)

# Compute the dot product and print it
print("Dot product of x and y is %s" % d)

# Represent the elementwise product of x and y
z = x.element_prod(y) # or x.element_mul(y)

# Compute and print the result
print("Elementwise product of x and y is %s" % z)

# Represent the outer product of x and y
c = x.outer(y)

# Compute and print the result
print("Outer product of x and y:\n%s" % c)


#
# Matrix and matrix-vector products
#

# Represent the elementwise product of a and b
c = a.element_prod(b) # or a.elementwise_mul(b)

# Compute and print the result
print("Elementwise product of a and b:\n%s" % c)

# Represent the matrix product of a and b
c = a * b

# Compute and print the result
print("Matrix product of a and b:\n%s" % c)

# Represent the matrix-vector product of a and x
c = a * x

# Compute and print the result
print("Matrix-vector product of a and x:\n%s" % c)



