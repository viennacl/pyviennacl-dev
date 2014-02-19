#!python

"""
In this example, we investigate the construction and basic usage of
PyViennaCL's dense matrix (Matrix) and Vector types, and discuss some
important issues about integration with Python and NumPy data types, and
PyViennaCL's computational architecture.

If you are familiar with NumPy, you might need about 5 minutes to grasp the
content here. If you are not so familiar, you might need 10 minutes.
"""

# Import PyViennaCL and NumPy
import pyviennacl as p
import numpy as np

# Create our datastructures on the host
x = [1.0, 2.0, 3.0, 4.0, 5.0] # We can create PyViennaCL Vectors from lists
a = np.array([[1.0, 2.0, 3.0],
              [0.0, 3.0, 4.0],
              [0.0, 0.0, 5.0]]) # We can create PyViennaCL Matrices from arrays

# Create corresponding ViennaCL datastructures on the compute device
y = p.Vector(x)
b = p.Matrix(a) # This is a dense matrix

# Copy the data back to the host and check that it's equal
z = y.value # z is now a 1-D numpy array with dtype float64
c = b.value # c is now a 2-D numpy array with dtype float64

if (z == x).all() and (c == a).all():
    print("Successfully transferred data to and from the compute device!")

# We can modify elements of the ViennaCL device structures, but since this 
# incurs a compute kernel initialisation and buffer transfer, it is very slow!
y[0] = 0.0
b[2, 2] = -1.0

x[0] = 0.0     # We should also modify our original data to keep track..
a[2, 2] = -1.0

# And we can do comparisons seamlessly between NumPy and PyViennaCL types!
if (a == b).all() and (x == y).all():
    print("Successfully modified data structures on host and device!")

# We also need to be sure that we are consistent with respect to the data-types
# we use. For instance, we should not mix integer and floating point types.
#
# By default, PyViennaCL objects inherit the dtype of the objects from which
# they are created, or (if that is ambiguous), float64.
print("a and b have dtypes of %s and %s" %
      (np.result_type(a), np.result_type(b)))

# PyViennaCL makes an effort to convert objects to the appropriate dtype where
# dtypes have been mixed, but this is often ambiguous and unpredictable, and
# so it is recommended that users make efforts to keep they dtypes consistent.
i = 1L # Create a long integer
print("i has dtype of %s" % (np.result_type(i)))
y[0] = i # Remember, this sort of elementwise assignation is *very slow*!
print("y has values %s and dtype %s" %
      (y, np.result_type(y)))

# And, of course, we can perform basic arithemetic operations with PyViennaCL,
# mixing native Python types with NumPy and PyViennaCL types:
z = (x + y + z) / 2.0
print("z is now of type %s, dtype %s, and with values %s" 
      % (type(z), np.result_type(z), z))
# Notice that z has `Div' type. This is because the z object represents the
# arithmetic expression `(x + y + z) / 2.0', and this is only computed when
# the result is needed, in order to maximise performance.

# And we can do less basic arithmetic!
print("The sine of the values of z is %s" %
      p.sin(z))
# PyViennaCL exposes many elementwise mathematical functions.
# See help(p.math) for more information. 
