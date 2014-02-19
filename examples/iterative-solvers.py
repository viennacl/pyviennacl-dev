#!python

"""
PyViennaCL provides iterative solvers for sparse and dense linear systems.
The API is documented in ``help(pyviennacl.linalg.solve)``. In particular,
the solver to use is determined by the tag instance supplied to the ``solve``
function.

The iterative solvers have various parameters for tuning the error tolerance,
and various requirements for the form of the system matrix, as described in the
documentation for the corresponding tag classes.

For this reason, we only demonstrate here the use of the GMRES solver for a
general system.
"""

import pyviennacl as p
import numpy as np
import os, random
from util import read_mtx, read_vector

A = read_mtx(os.path.join(os.path.dirname(os.path.realpath(__file__)), "mat65k.mtx"),
             dtype=np.float32)
print("Loaded system matrix")

b = read_vector(os.path.join(os.path.dirname(os.path.realpath(__file__)), "rhs65025.txt"),
                dtype=np.float32)
print("Loaded RHS vector")

# Construct the tag to denote the GMRES solver
tag = p.gmres_tag(tolerance = 1e-5, max_iterations = 150, krylov_dim = 50)

# Solve the system
x = p.solve(A, b, tag)

# Show some info
print("Num. iterations: %s" % tag.iters)
print("Estimated error: %s" % tag.error)

