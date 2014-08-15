pycore: core PyViennaCL features
--------------------------------

.. module:: pyviennacl.pycore

This submodule contains PyViennaCL's core functionality, including
types for representing and manipulating scalars, vectors and matrices
on the host and compute device, with a variety of numerical data types
(equivalent to the NumPy concept of ``dtype``).

Also provided are routines for type conversion, arithmetic and linear
algebra, to BLAS level 3. Vector and matrix types can be sensibly
converted to and from NumPy arrays, and sparse types can be converted
to and from SciPy sparse types.

Finally, support for creating custom operations integrated into the
PyViennaCL expression graph is provided here in the
:class:`CustomNode` class.

Background information
----------------------

Because in heterogeneous computing systems copying data across the bus
from host memory to device memory (or vice versa) commonly incurs a
proportionally substantial wait, PyViennaCL adopts a policy of delayed
execution.  Arithmetical expressions are represented as a binary tree,
and are only dispatched to be computed when the result of the
computation is required, such as for output, or when the computation
is explicitly executed.

Thus, the result of adding two :class:`Matrix` objects is not another
``Matrix`` object, but an :class:`Add` object, which is converted to a
``Matrix`` when the result is accessed.  Consequently, this submodule
provides a number of classes for elementary operations, such as
``Add``, for representation in an expression tree. Each of these
expression tree classes is a subclass of :class:`Node` type, with
``Node`` providing basic functionality for the construction of the
expression tree.

In the language of PyViennaCL, data classes such as :class:`Scalar`,
:class:`Vector` and `Matrix` constitute leaves on the expression tree,
and as such, each of these data classes inherits from the
:class:`Leaf` type, which provides general functionality for leaf
construction.

Importantly, you can treat any ``Node`` object -- such as an ``Add``
instance -- as if it were a ``Leaf`` object explicitly representing
some data. This means that you can express mathematical operations
using ``Node`` and ``Leaf`` instances transparently.

``Node`` and ``Leaf`` instances are flattened into a
:class:`Statement` object when the expression is executed. The
``Statement`` class recursively constructs the C++ object equivalent
to the expression tree as represented in Python, and this is then
dispatched to the compute device. The result is cached so that
multiple identical computations are not made.

On object construction and access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the same reasons of bus and compute latency, PyViennaCL does not
encourage the elementwise construction of ``Matrix`` or ``Vector``
objects, or the accessing of individual scalar elements from any such
type; the waits incurred make such access painfully slow.

Instead, you should construct dense matrices and vectors using
preconstructed types: NumPy ``array`` objects can be supplied to
construct both matrices and vectors -- as long as the arrays are of
the correct dimensionality -- and Python lists can be supplied to
construct vectors, as long as the ``dtype`` of the list is
comprehensible. In both the list case and the array case, you can use
Python or NumPy numeric data types: usage of the NumPy ``dtype`` is
recommended, since this is more explicit.

Elementwise accesses to array-like PyViennaCL types incur the
computation of any expression, the transfer of the result to the host,
the representation of that result as a NumPy ``ndarray`` (which may
incur a large memory cost), and then the accessing of the element from
that array.

The exception to this rule is the set of PyViennaCL sparse matrix
types, which do support elementwise construction and access, because
they are backed by a transparent host-memory cache which is only
flushed to and from the device when necessary, in the manner of the
delayed execution described above.

To force the execution of an expression or the flushing of a matrix,
access the ``result`` attribute, which will give a ``Leaf`` object. To
retrieve a NumPy ``array`` containing the data of the PyViennaCL
``Leaf`` or ``Node``, use the ``as_ndarray()`` method; there is an
equivalent method for sparse types. If you are not particularly
concerned about the type of object you retrieve, access the ``value``
attribute: for PyViennaCL scalars, this provides a NumPy / Python
scalar object; for vectors and matrices, it provides an appropriate
NumPy ``array``; and for sparse matrices, where SciPy is installed, it
provides a SciPy ``lil_matrix``.

The relationship of an object's ``result`` to its ``value`` is the
same as that of compute device to host memory.

Ranges and slices of matrices and vectors are well supported,
including the assignment of one matrix to a sub-matrix of another, as
long as the matrix dimensions are commensurable. For instance::

  >>> a[5:10, 5:10] = b                                  # doctest: +SKIP 

Submodule contents
------------------

Leaf types
^^^^^^^^^^
========================= =================================================
:class:`HostScalar`       Represents a scalar in host memory
:class:`Scalar`           Represents a scalar in compute device memory
:class:`Vector`           Represents a vector in compute device memory
:class:`CompressedMatrix` Represents a sparse matrix with compressed-row
                          storage in compute device memory
:class:`CoordinateMatrix` Represents a sparse matrix with a coordinate
                          storage format in compute device memory
:class:`ELLMatrix`        Represents a sparse matrix with an ELL storage
                          format in compute device memory
:class:`HybridMatrix`     Represents a sparse matrix with a hybrid storage
                          format, combining ELL and compressed-row storage,
                          in compute device memory
:class:`Matrix`           Represents a dense matrix, with either row-major
                          (default) or column-major storage.
========================= =================================================

Supported numeric data types:
  int8, int16, int32, int64,
  uint8, uint16, uint32, uint64,
  float32, float64.

Many operations are only currently supported using a floating poit numeric
data type, but wider numeric support is forthcoming in later versions.

Node types
^^^^^^^^^^
===================== ====================================================
:class:`Norm_1`       Order-1 norm
:class:`Norm_2`       Order-2 norm
:class:`Norm_Inf`     Infinity norm
:class:`ElementAbs`   Elementwise abs
:class:`ElementAcos`  Elementwise acos
:class:`ElementAsin`  Elementwise asin
:class:`ElementAtan`  Elementwise atan
:class:`ElementCeil`  Elementwise ceil
:class:`ElementCos`   Elementwise cos
:class:`ElementCosh`  Elementwise cosh
:class:`ElementExp`   Elementwise exp
:class:`ElementFabs`  Elementwise fabs
:class:`ElementFloor` Elementwise floor
:class:`ElementLog`   Elementwise log
:class:`ElementLog10` Elementwise log10
:class:`ElementSin`   Elementwise sin
:class:`ElementSinh`  Elementwise sinh
:class:`ElementSqrt`  Elementwise sqrt
:class:`ElementTan`   Elementwise tan
:class:`ElementTanh`  Elementwise tanh
:class:`Trans`        Matrix transpose
:class:`Assign`       Assign (copy) the values of one object to another of
                      the same type. You can Assign across different matrix
                      layouts.
:class:`InplaceAdd`   In-place addition
:class:`InplaceSub`   In-place subtraction
:class:`Add`          Addition (allocates returns a new object)
:class:`Sub`          Subtraction (allocates returns a new object)
:class:`Mul`          Multiplication:

                      * Scalar by scalar -> scalar;
                      * scalar by vector -> scaled vector;
                      * scalar by matrix -> scaled matrix;
                      * vector by vector -> undefined;
                      * vector by matrix -> undefined;
                      * matrix by vector -> matrix-vector product;
                      * matrix by matrix -> matrix-matrix product.

                      The concern in defining these semantics has been to
                      preserve the dimensionality of the operands in the
                      result.

                      The Mul class does not map directly onto the * operator
                      for every class.
:class:`Div`          Scalar division
:class:`ElementPow`   Elementwise exponentiation
:class:`ElementProd`  Elementwise multiplication
:class:`ElementDiv`   Elementwise division
:class:`Dot`          Inner (dot) product of two vectors
===================== ====================================================

Most of these expression classes are implicitly constructed by
arithmetical convenience functions, including the standard Python
arithmetic operators.  For instance, for two commensurate objects
``a`` and ``b``::

  >>> ((a + b) == p.Add(a, b)).all()                     # doctest: +SKIP
  True

For more information about the semantics of the arithmetic operators,
such as *, +, -, **, / and //, see the docstrings for the individual
classes involved; for the default semantics, see the docstrings for
the :class:`MagicMethods` class.

The equality operator falls back to NumPy's ``equal`` function for all
classes that are not scalars, or that do not produce scalars as a
result; note that this incurs a copy of the object into host
memory. In the scalar case, simple numerical equality is used.

Submodule reference
-------------------

.. autoclass:: MagicMethods
   :private-members:
   :special-members:

Leaf types
^^^^^^^^^^

.. autoclass:: Leaf
.. autoclass:: ScalarBase
.. autoclass:: HostScalar
.. autoclass:: Scalar
.. autoclass:: Vector
.. autoclass:: Matrix
.. autoclass:: SparseMatrixBase
.. autoclass:: CompressedMatrix
.. autoclass:: CoordinateMatrix
.. autoclass:: ELLMatrix
.. autoclass:: HybridMatrix

Node types
^^^^^^^^^^

.. autoclass:: Node
.. autoclass:: CustomNode
.. autoclass:: Norm_1
.. autoclass:: Norm_2
.. autoclass:: Norm_Inf
.. autoclass:: Neg
.. autoclass:: ElementAbs
.. autoclass:: ElementAcos
.. autoclass:: ElementAsin
.. autoclass:: ElementAtan
.. autoclass:: ElementCeil
.. autoclass:: ElementCos
.. autoclass:: ElementCosh
.. autoclass:: ElementExp
.. autoclass:: ElementFabs
.. autoclass:: ElementFloor
.. autoclass:: ElementLog
.. autoclass:: ElementLog10
.. autoclass:: ElementSin
.. autoclass:: ElementSinh
.. autoclass:: ElementSqrt
.. autoclass:: ElementTan
.. autoclass:: ElementTanh
.. autoclass:: Trans
.. autoclass:: Assign
.. autoclass:: InplaceAdd
.. autoclass:: InplaceSub
.. autoclass:: Add
.. autoclass:: Sub
.. autoclass:: Mul
.. autoclass:: Div
.. autoclass:: ElementPow
.. autoclass:: ElementProd
.. autoclass:: ElementDiv
.. autoclass:: Dot

.. autoclass:: Statement

