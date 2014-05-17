"""PyViennaCL
==========

This extension provides the Python bindings for the `ViennaCL
<http://viennacl.sourceforge.net/>`_ linear algebra and numerical
computation library for GPGPU and heterogeneous systems. `ViennaCL
<http://viennacl.sourceforge.net/>`_ itself is a header-only C++
library, so these bindings make available to Python programmers
ViennaCL's fast OpenCL and CUDA algorithms, in a way that is idiomatic
and compatible with the Python community's most popular scientific
packages, NumPy and SciPy.

PyViennaCL is divided into five submodules, of which four are designed
for direct use by users:

  * :doc:`pycore`: user-friendly classes for representing the main ViennaCL
    objects, such as Matrix or Vector;
  * :doc:`linalg`: an explicit interface to a number of ViennaCL's linear
    algebra routines, such as matrix solvers and eigenvalue computation;
  * :doc:`vclmath`: convenience functions akin to the standard math module
    (you can also access it as ``pyviennacl.math``);
  * :doc:`util`: utility functions, such as to construct an appropriate
    ViennaCL object from an ndarray (Matrix or Vector), or to provide basic
    debug logging;
  * _viennacl: a raw C++ interface to ViennaCL, with no stable API.

Nonetheless, all of PyViennaCL's functionality is available from the
top-level pyviennacl namespace. So, if you want help on the Matrix class,
you can just run::

  >>> import pyviennacl as p
  >>> help(p.Matrix)                                     # doctest: +SKIP

For help on PyViennaCL's core functionality in general, or
PyViennaCL's high-level linear algebra functions, run::

   >>> help(p.pycore)                                    # doctest: +SKIP

or::

   >>> help(p.linalg)                                    # doctest: +SKIP

"""

from .version import VERSION_TEXT as __version__
from ._viennacl import __version__ as __viennacl_version__

from .pycore import *
from .linalg import *
from .util import *
from .vclmath import *

from . import vclmath as math

# TODO: __all__
