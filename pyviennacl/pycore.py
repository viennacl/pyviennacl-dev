"""
This submodule contains PyViennaCL's core functionality, including types for
representing and manipulating scalars, vectors and matrices on the host and
compute device, with a variety of numerical data types (equivalent to the
NumPy concept of ``dtype``).

Also provided are routines for type conversion, arithmetic and linear
algebra, to BLAS level 3. Vector and matrix types can be sensibly converted
to and from NumPy arrays, and support for SciPy sparse matrix types is forth-
coming.

Background information
----------------------
Because in heterogeneous computing systems copying data across the bus from
host memory to device memory (or vice versa) commonly incurs a proportionally
substantial wait, PyViennaCL adopts a policy of delayed execution.
Arithmetical expressions are represented as a binary tree, and are only
dispatched to be computed when the result of the computation is necessary.

Thus, the result of adding two Matrix objects is not another Matrix object,
but an Add object, which is converted to a Matrix when the result is accessed.
Consequently, this submodule provides a number of classes for elementary
arithmetical operations, such as Add, for representation in an expression
tree. Each of these expression tree classes is a subclass of Node type, with
Node providing basic functionality for the construction of the expression tree.

In the language of ViennaCL, 'data' classes such as Scalar, Vector and
Matrix constitute leaves on the expression tree, and as such, each of these
data classes inherits from the Leaf type, which provides general functionality
for leaf construction.

Node and Leaf instances are flattened into a Statement object when the
expression is executed. The Statement class recursively constructs the C++
object equivalent to the expression tree as represented in Python, and this is
then dispatched to the compute device. The result is cached so that multiple
identical computations are not made.

On object construction and access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the same reasons of bus and compute latency, PyViennaCL does not support
the elementwise construction of Matrix or Vector objects, or the accessing of
individual scalar elements from any such type; the waits incurred make such
access painfully slow.

Instead, you must construct dense matrices and vectors using preconstructed
types: NumPy ``array``s can be supplied to construct both matrices and
vectors -- as long as the arrays are of the correct dimensionality -- and
Python lists can be supplied to construct vectors, as long as the ``dtype``
of the list is comprehensible. Construction from lists, but not arrays, is
supported if the element type is a PyViennaCL scalar type. In both the list
case and the array case, you can use Python or NumPy numeric data types:
NumPy ``dtype``s are recommended, since these are more explicit.

Elementwise accesses to array-like PyViennaCL types incur the computation of
any expression, the transfer of the result to the host, the representation of
that result as a NumPy ``ndarray`` (which may incur a large memory cost), and
then the accessing of the element from that array.

The exception to this rule is the set of PyViennaCL sparse matrix types, which
do support elementwise construction and access, because they are backed by a
transparent host-memory cache which is only flushed to and from the device
when necessary, in the manner of the delayed execution described above.

To force the execution of an expression or the flushing of a matrix, access
the ``result`` attribute. To retrieve a NumPy ``array`` containing the data
of the PyViennaCL Leaf or Node, use the ``as_ndarray()`` method. If you are
not particularly concerned about the type of object you retrieve, access the
``value`` attribute: for PyViennaCL scalars, this provides a NumPy / Python
scalar object; for vectors and matrices, it provides an appropriate NumPy
``array``.

Ranges and slices of matrices and vectors are well supported, including
the assignment of one matrix to a sub-matrix of another, as long as the matrix
dimensions are commensurable. For instance::

  >>> a[5:10, 5:10] = b                                  # doctest: +SKIP 

Submodule contents
------------------

Leaf types
^^^^^^^^^^
================ ==================================
HostScalar       Represents a scalar in host memory
Scalar           Represents a scalar in compute device memory
Vector           Represents a vector in compute device memory
CompressedMatrix Represents a sparse matrix with compressed-row storage
                 in compute device memory
CoordinateMatrix Represents a sparse matrix with a coordinate storage format
                 in compute device memory
ELLMatrix        Represents a sparse matrix with an ELL storage format in
                 compute device memory
HybridMatrix     Represents a sparse matrix with a hybrid storage format,
                 combining ELL and compressed-row storage, in compute device
                 memory
Matrix           Represents a dense matrix, with either row-major (default) or
                 column-major storage.
================ ==================================

Supported numeric data types:
  int8, int16, int32, int64,
  uint8, uint16, uint32, uint64,
  float16, float32, float64.

Many operations are only currently supported using a floating poit numeric
data type, but wider numeric support is forthcoming in later versions.

Node types
^^^^^^^^^^
============ =============================================================
Norm_1       Order-1 norm
Norm_2       Order-2 norm
Norm_Inf     Infinity norm
ElementAbs   Elementwise abs
ElementAcos  Elementwise acos
ElementAsin  Elementwise asin
ElementAtan  Elementwise atan
ElementCeil  Elementwise ceil
ElementCos   Elementwise cos
ElementCosh  Elementwise cosh
ElementExp   Elementwise exp
ElementFabs  Elementwise fabs
ElementFloor Elementwise floor
ElementLog   Elementwise log
ElementLog10 Elementwise log10
ElementSin   Elementwise sin
ElementSinh  Elementwise sinh
ElementSqrt  Elementwise sqrt
ElementTan   Elementwise tan
ElementTanh  Elementwise tanh
Trans        Matrix transpose
Assign       Assign (copy) the values of one object to another of the same
             type. You can Assign across different matrix layouts.
InplaceAdd   In-place addition
InplaceSub   In-place subtraction
Add          Addition (allocates returns a new object)
Sub          Subtraction (allocates returns a new object)
Mul          Multiplication:
             * Scalar by scalar -> scalar;
             * scalar by vector -> scaled vector;
             * scalar by matrix -> scaled matrix;
             * vector by vector -> undefined;
             * vector by matrix -> undefined;
             * matrix by vector -> matrix-vector product;
             * matrix by matrix -> matrix-matrix product.
             The concern in defining these semantics has been to preserve
             the dimensionality of the operands in the result.
             The Mul class does not map directly onto the * operator for
             every class.
Div          Scalar division
ElementProd  Elementwise scalar multiplication
ElementDiv   Elementwise scalar division
Dot          Inner (dot) product of two vectors
============ =============================================================

Most of these expression classes are implicitly constructed by arithmetical
convenience functions, including the standard Python arithmetic operators.
For instance, for two commensurate objects ``a`` and ``b``::

  >>> ((a + b) == p.Add(a, b)).all()                     # doctest: +SKIP
  True

For more information about the semantics of the arithmetic operators, such as
*, +, -, **, / and //, see the docstrings for the individual classes involved;
for the default semantics, see the docstrings for the ``MagicMethods`` class.

The equality operator falls back to NumPy's ``equal`` function for all classes
that are not scalars, or that do not produce scalars as a result; in the
scalar case, simple numerical equality is used.
"""

from __future__ import division
import logging, math
from pyviennacl import (_viennacl as _v,
                        util)
from numpy import (ndarray, array, zeros,
                   inf, nan, dtype,
                   equal as np_equal, array_equal,
                   result_type as np_result_type,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)

try:
    from scipy import sparse
    WITH_SCIPY = True
except:
    WITH_SCIPY = False

log = logging.getLogger(__name__)

# This dict maps ViennaCL container subtypes onto the strings used for them
vcl_statement_node_subtype_strings = {
    _v.statement_node_subtype.INVALID_SUBTYPE: 'node',
    _v.statement_node_subtype.HOST_SCALAR_TYPE: 'host',
    _v.statement_node_subtype.DEVICE_SCALAR_TYPE: 'scalar',
    _v.statement_node_subtype.DENSE_VECTOR_TYPE: 'vector',
    _v.statement_node_subtype.IMPLICIT_VECTOR_TYPE: 'implicit_vector',
    _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE: 'matrix_row',
    _v.statement_node_subtype.DENSE_COL_MATRIX_TYPE: 'matrix_col',
    _v.statement_node_subtype.IMPLICIT_MATRIX_TYPE: 'implicit_matrix',
    _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE: 'compressed_matrix',
    _v.statement_node_subtype.COORDINATE_MATRIX_TYPE: 'coordinate_matrix',
    _v.statement_node_subtype.ELL_MATRIX_TYPE: 'ell_matrix',
    _v.statement_node_subtype.HYB_MATRIX_TYPE: 'hyb_matrix'
}

# This dict maps ViennaCL numeric types onto the C++ strings used for them
vcl_statement_node_numeric_type_strings = {
    _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE: 'index',
    _v.statement_node_numeric_type.CHAR_TYPE: 'char',
    _v.statement_node_numeric_type.UCHAR_TYPE: 'uchar',
    _v.statement_node_numeric_type.SHORT_TYPE: 'short',
    _v.statement_node_numeric_type.USHORT_TYPE: 'ushort',
    _v.statement_node_numeric_type.INT_TYPE: 'int',
    _v.statement_node_numeric_type.UINT_TYPE: 'uint',
    _v.statement_node_numeric_type.LONG_TYPE: 'long',
    _v.statement_node_numeric_type.ULONG_TYPE: 'ulong',
    _v.statement_node_numeric_type.HALF_TYPE: 'half',
    _v.statement_node_numeric_type.FLOAT_TYPE: 'float',
    _v.statement_node_numeric_type.DOUBLE_TYPE: 'double',
}

# This dict is used to map NumPy dtypes onto OpenCL/ViennaCL numeric types
HostScalarTypes = {
    'int8': _v.statement_node_numeric_type.CHAR_TYPE,
    'int16': _v.statement_node_numeric_type.SHORT_TYPE,
    'int32': _v.statement_node_numeric_type.INT_TYPE,
    'int64': _v.statement_node_numeric_type.LONG_TYPE,
    'uint8': _v.statement_node_numeric_type.UCHAR_TYPE,
    'uint16': _v.statement_node_numeric_type.USHORT_TYPE,
    'uint32': _v.statement_node_numeric_type.UINT_TYPE,
    'uint64': _v.statement_node_numeric_type.ULONG_TYPE,
    'float16': _v.statement_node_numeric_type.HALF_TYPE,
    'float32': _v.statement_node_numeric_type.FLOAT_TYPE,
    'float64': _v.statement_node_numeric_type.DOUBLE_TYPE,
    'float': _v.statement_node_numeric_type.DOUBLE_TYPE
}

# Constants for choosing matrix storage layout
ROW_MAJOR = 1
COL_MAJOR = 2

vcl_layout_strings = {
    ROW_MAJOR: 'row',
    COL_MAJOR: 'col'
}


class NoResult(object): 
    """
    This no-op class is used to represent when some ViennaCL operation produces
    no explicit result, aside from any effects it may have on the operands.
    
    For instance, in-place operations can return NoResult, as can Assign.
    """
    pass


class MagicMethods(object):
    """
    A class to provide convenience methods for arithmetic and BLAS access.

    Classes derived from this will inherit lots of useful features applicable
    to PyViennaCL. For more information, see the individual methods below.
    """
    flushed = False

    def result_container_type(self):
        """
        This function should be overridden, with the following semantics.

        Parameters
        ----------
        None

        Returns
        -------
        x : type
            The type that the operation or object represented by an instance
            of this class should return as a result on execution.

        Raises
        ------
        NotImplementedError
            If you do not override this function in a class derived from
            MagicMethods.
        """
        raise NotImplementedError("Why is this happening to you?")

    def copy(self):
        """
        Returns a new instance of this class representing a new copy of this
        instance's data.
        """
        return type(self)(self)

    def norm(self, ord=None):
        """
        Returns a norm of this instance, if that is defined.
        
        The norm returned depends on the ``ord`` parameter, as in SciPy.
        * If this instance is a ``Matrix``, then ``ord`` must be ``None``,
          and the only norm supported is the Frobenius norm.

        Parameters
        ----------
        ord : {1, 2, inf, 'fro', None}
            Order of the norm.
            inf means NumPy's ``inf`` object.
            'fro' means the string 'fro', and denotes the Frobenius norm.
            If None and self is a Matrix instance, then assumes 'fro'.
        """
        if ord == 1:
            #return Norm_1(self) TODO NOT WORKING WITH SCHEDULER
            return Scalar(_v.norm_1(self.vcl_leaf),
                          dtype = self.dtype)
        elif ord == 2:
            #return Norm_2(self)
            return Scalar(_v.norm_2(self.vcl_leaf),
                          dtype = self.dtype)
        elif ord == inf:
            #return Norm_Inf(self)
            return Scalar(_v.norm_inf(self.vcl_leaf),
                          dtype = self.dtype)
        else:
            return Scalar(_v.norm_frobenius(self.vcl_leaf),
                          dtype = self.dtype)

    #@property
    #def norm_1(self):
    #    return Norm_1(self) #.result

    #@property
    #def norm_2(self):
    #    return Norm_2(self) #.result

    #@property
    #def norm_inf(self):
    #    return Norm_Inf(self) #.result

    def prod(self, rhs):
        """
        Returns ``(self * rhs)``.
        """
        return (self * rhs)

    def element_prod(self, rhs):
        """
        Returns the elementwise product of ``self`` and ``rhs``, for some
        ``rhs`` (right-hand side).
        """
        return ElementProd(self, rhs)
    element_mul = element_prod

    def element_div(self, rhs):
        """
        Returns the elementwise division of ``self`` and ``rhs``, for some
        ``rhs`` (right-hand side).
        """
        return ElementDiv(self, rhs)

    def __pow__(self, rhs):
        """
        x.__pow__(y) <==> x**y

        Notes
        -----
        For array-like types, this is computed elementwise. But ViennaCL does
        not currently support elementwise exponentiation in the scheduler, so
        this incurs the computation of the expression represented by ``y`` at
        this point. Nonetheless, the result is the appropriate PyViennaCL type.
        """
        if isinstance(rhs, MagicMethods):
            if not self.shape == rhs.shape:
                raise TypeError("Operands must have the same shape!")
            return self.result_container_type(_v.element_pow(self.vcl_leaf,
                                                             rhs.vcl_leaf),
                                              dtype = self.dtype,
                                              layout = self.layout)
        else:
            return self.result_container_type(self.value ** rhs,
                                              dtype = self.dtype,
                                              layout = self.layout)

    def __rpow__(self, rhs):
        """
        x.__rpow__(y) <==> y**x
        """
        if isinstance(rhs, MagicMethods):
            if not self.shape == rhs.shape:
                raise TypeError("Operands must have the same shape!")
            return self.result_container_type(_v.element_pow(rhs.vcl_leaf,
                                                             self.vcl_leaf),
                                              dtype = self.dtype)
        else:
            return self.result_container_type(rhs ** self.value,
                                              dtype = self.dtype)
        
    def __eq__(self, rhs):
        """
        The equality operator.

        Parameters
        ----------
        rhs : {scalar, Vector, Matrix, ndarray, etc}
            Comparator

        Returns
        -------
        retval : {bool, ndarray}
            If the r.h.s. is elementwise comparable with a Vector, Matrix or
            ndarray, then an array of boolean values is returned; see NumPy's
            ``equal`` function. If the r.h.s. is a scalar, then a boolean
            value is returned. Otherwise, the behaviour is undefined, but the
            Python ``==`` operator is used to compare the ``result`` attribute
            of ``self`` to the r.h.s., in an attempt at meaningfulness.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.value == rhs.value
            else:
                return self.value == rhs
        if self.flushed:
            if isinstance(rhs, MagicMethods):
                return np_equal(self.as_ndarray(), rhs.as_ndarray())
            elif isinstance(rhs, ndarray):
                return np_equal(self.as_ndarray(), rhs)
            else:
                return self.value == rhs
        else:
            return self.result == rhs

    def __ne__(self, rhs):
        return not (self == rhs)

    def __hash__(self):
        ## TODO implement better hash function
        return object.__hash__(self)
        #return id(self)

    def __contains__(self, item):
        return (item in self.as_ndarray())

    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

    def __add__(self, rhs):
        """
        x.__add__(y) <==> x+y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value + rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.value + rhs
        op = Add(self, rhs)
        return op

    def __sub__(self, rhs):
        """
        x.__sub__(y) <==> x-y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value - rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.value - rhs
        op = Sub(self, rhs)
        return op

    def __mul__(self, rhs):
        """
        x.__mul__(y) <==> x*y

        Returns
        -------
        z : {Mul(x, y), (x.value * rhs)}
            Returns a Mul instance if defined.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value * rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.value * rhs
        op = Mul(self, rhs)
        return op

    def __floordiv__(self, rhs):
        """
        x.__floordiv__(y) <==> x//y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value // rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.value // rhs
        op = math.floor(Div(self, rhs))
        return op

    def __truediv__(self, rhs):
        """
        x.__truediv__(y) <==> x/y

        Notes
        -----
        PyViennaCL automatically adopts Python 3.0 division semantics, so the
        ``/`` division operator is never floor (integer) division, and always
        true floating point division.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value / rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.value / rhs
        op = Div(self, rhs)
        return op
    __div__ = __truediv__

    def __iadd__(self, rhs):
        """
        x.__iadd__(y) <==> x+=y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    self.value += rhs.value
            else:
                self.value += rhs
            return self
        if isinstance(self, Node):
            return Add(self, rhs)
        else:
            op = InplaceAdd(self, rhs)
            op.execute()
            return self

    def __isub__(self, rhs):
        """
        x.__isub__(y) <==> x-=y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    self.value -= rhs.value
            else:
                self.value -= rhs
            return self
        if isinstance(self, Node):
            return Sub(self, rhs)
        else:
            op = InplaceSub(self, rhs)
            op.execute()
            return self

    def __imul__(self, rhs):
        """
        x.__imul__(y) <==> x*=y
       
        Notes
        -----
        See the definition of the ``*`` operator for more information about
        the semantics of ``*=``.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    self.value *= rhs.value
            else:
                self.value *= rhs
            return self
        if isinstance(self, Node):
            return Mul(self, rhs)
        else:
            op = Mul(self, rhs)
            return op.result

    def __ifloordiv__(self, rhs):
        """
        x.__ifloordiv__(y) <==> x//=y
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    self.value //= rhs.value
            else:
                self.value //= rhs
            return self
        if isinstance(self, Node):
            return math.floor(Div(self, rhs))
        else:
            op = math.floor(Div(self, rhs))
            return op.result

    def __itruediv__(self, rhs):
        """
        x.__itruediv__(y) <==> x/=y

        Notes
        -----
        PyViennaCL automatically adopts Python 3.0 division semantics, so the
        ``/`` division operator is never floor (integer) division, and always
        true floating point division.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    self.value /= rhs.value
            else:
                self.value /= rhs
            return self
        if isinstance(self, Node):
            return Div(self, rhs)
        else:
            op = Div(self, rhs)
            return op.result

    def __radd__(self, rhs):
        """
        x.__radd__(y) <==> y+x

        Notes
        -----
        Addition is commutative.
        """
        return self + rhs

    def __rsub__(self, rhs):
        """
        x.__rsub__(y) <==> y-x
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value - self.value,
                                                 dtype = rhs.dtype)
            return rhs - self
        return rhs - self.value

    def __rmul__(self, rhs):
        """
        x.__rmul__(y) <==> y*x

        Notes
        -----
        Multiplication is commutative.
        """
        return self * rhs

    def __rfloordiv__(self, rhs):
        """
        x.__rfloordiv__(y) <==> y//x
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value // self.value,
                                                 dtype = rhs.dtype)
            return rhs // self
        return rhs // self.value

    def __rtruediv__(self, rhs):
        """
        x.__rtruediv__(y) <==> y/x

        Notes
        -----
        PyViennaCL automatically adopts Python 3.0 division semantics, so the
        ``/`` division operator is never floor (integer) division, and always
        true floating point division.
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value / self.value,
                                                 dtype = rhs.dtype)
            return rhs / self
        return rhs / self.value
    __rdiv__ = __rtruediv__

    def __neg__(self):
        """
        x.__neg__() <==> -x
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(-self.value, dtype = self.dtype)
        return Mul(self.dtype.type(-1), self)

    def __abs__(self):
        """
        x.__abs__() <==> abs(x)

        Notes
        -----
        OpenCL does not provide for ``abs`` on floating point types, so if
        your instance has a floating point data type, ``abs(x)`` is equivalent
        to ``fabs(x)``.

        On array-like types, this is computed elementwise.
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(abs(self.value),
                                              dtype = self.dtype)
        elif issubclass(np_result_type(self).type, float):
            # No floating abs in OpenCL
            return ElementFabs(self)
        else:
            return ElementAbs(self)

    def __floor__(self):
        """
        x.__floor__() <==> math.floor(self)

        Notes
        -----
        On array-like types, this is computed elementwise.
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(math.floor(self.value),
                                              dtype = self.dtype)
        return ElementFloor(self)

    def __ceil__(self):
        """
        x.__ceil__() <==> math.ceil(self)
        
        Notes
        -----
        On array-like types, this is computed elementwise.
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(math.ceil(self.value),
                                              dtype = self.dtype)
        return ElementCeil(self)


class View(object):
    """
    This class represents a C++ ViennaCL range or slice, as a 'view' on an
    object. A View instance is object-independent; i.e., it represents an
    abstract view.
    """
    start = None
    stop = None
    step = None

    def __init__(self, key, axis_size):
        """
        Construct a View object.

        Parameters
        ----------
        key : slice
        axis_size : int
            The number of elements along the axis of which the instance
            of this class is a view.
        """
        start, stop, step = key.indices(axis_size)

        if step == 1:
            # then range -- or slice!
            self.vcl_view = _v.slice(start, 1, (stop-start))
        else:
            # then slice
            self.vcl_view = _v.slice(start, step,
                                     int(math.ceil((stop-start)/step)))

        self.slice = key
        self.start = start
        self.stop = stop
        self.step = step


class Leaf(MagicMethods):
    """
    This is the base class for all ``leaves`` in the ViennaCL expression tree
    system. A leaf is any type that can store data for use in an operation,
    such as a scalar, a vector, or a matrix.
    """
    shape = None # No shape yet -- not even 0 dimensions
    flushed = True # Are host and device data synchronised?
    complexity = 0 # A leaf does not contribute computationally to a tree

    def __init__(self, *args, **kwargs):
        """
        Do initialisation tasks common to all Leaf subclasses, then pass
        control onto the overridden _init_leaf function.

        Tasks include expression computation and configuration of data types
        and views.
        """
        for arg in args:
            if isinstance(arg, list):
                for item in arg:
                    if isinstance(item, MagicMethods):
                        arg[arg.index(item)] = item.value

        if 'dtype' in kwargs.keys():    
            dt = dtype(kwargs['dtype']) 
            self.dtype = dt
        else:
            self.dtype = None
            
        if 'view_of' in kwargs.keys():
            self.view_of = kwargs['view_of']
        if 'view' in kwargs.keys():
            self.view = kwargs['view']

        self._init_leaf(args, kwargs)

    def __setitem__(self, key, value):
        if isinstance(value, Node):
            value = value.result
        ## TODO: This involves both a get and a set, if it works, so not very efficient..
        item = self[key] ## get
        if type(item) != type(value):
            if isinstance(item, ScalarBase):
                if isinstance(value, ScalarBase):
                    value = value.value
                if np_result_type(item) != np_result_type(value):
                    try:
                        value = np_result_type(item).type(value)
                    except:
                        log.exception("Failed to convert value dtype (%s) to item dtype (%s)" % 
                                      (np_result_type(item), np_result_type(value)))
                try:
                    try: # Assume matrix type
                        return self.vcl_leaf.set_entry(key[0], key[1], value) ## set
                    except: # Otherwise, assume vector
                        return self.vcl_leaf.set_entry(key, value) ## set
                except:
                    log.exception("Failed to set vcl entry")
            raise TypeError("Cannot assign %s to %s" % (type(value),
                                                        type(item)))
        if item.dtype != value.dtype:
            raise TypeError("Cannot assign across different dtypes! (%s and %s)" %
                            (item.dtype, value.dtype))
        if item.shape != value.shape:
            raise TypeError("Cannot assign across different shapes! (%s and %s)" %
                            (item.shape, value.shape))
        Assign(item, value).execute() ## set

    def _init_leaf(self, args, kwargs):
        """
        By default, leaf subclasses inherit a no-op further init function.
        
        If you're deriving from Leaf, then you probably want to override this,
        with the following semantics:

        Parameters
        ----------
        args : list
        kwargs : dict

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Unless overridden by a derived class.

        Notes
        -----
        The derived class should take ``args`` and ``kwargs`` and construct
        the leaf appropriately.
        """
        raise NotImplementedError("Help")

    def flush(self):
        """
        Override this function to implement caching functionality.
        """
        raise NotImplementedError("Should you be trying to flush this type?")

    @property
    def result_container_type(self):
        """
        The result_container_type for a leaf is always its own type.
        """
        return type(self)

    @property
    def result(self):
        """
        The result of an expression or subexpression consisting of a leaf is
        just the leaf itself.
        """
        return self

    def express(self, statement=""):
        """
        Construct a human-readable version of a ViennaCL expression tree
        statement from this leaf.
        """
        statement += type(self).__name__
        return statement

    def as_ndarray(self):
        """
        Return a NumPy ``ndarray`` containing the data within the underlying
        ViennaCL type.
        """
        return array(self.vcl_leaf.as_ndarray(), dtype=self.dtype)

    @property
    def value(self):
        """
        Return a NumPy ``ndarray`` containing the data within the underlying
        ViennaCL type.        
        """
        return self.as_ndarray()


class ScalarBase(Leaf):
    """
    This is the base class for all scalar types, regardless of their memory
    and backend context. It represents the dtype and the value of the scalar
    independently of one another.

    Because scalars are leaves in the ViennaCL expression graph, this class
    derives from the Leaf base class.
    """
    statement_node_type_family = _v.statement_node_type_family.SCALAR_TYPE_FAMILY
    ndim = 0 # Scalars are point-like, and thus 0-dimensional

    def _init_leaf(self, args, kwargs):
        """
        Do Scalar-specific initialisation tasks.
        1. Set the scalar value to the value given, or 0.
        2. If no dtype yet set, use the NumPy type promotion rules to deduce
           a dtype.
        """
        if 'value' in kwargs.keys():
            self._value = kwargs['value']
        elif len(args) > 0:
            if isinstance(args[0], ScalarBase):
                self._value = args[0].value
            else:
                self._value = args[0]
        else:
            self._value = 0

        if self.dtype is None:
            self.dtype = np_result_type(self._value)

        try:
            self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]
        except KeyError:
            raise TypeError("dtype %s not supported" % self.dtype.name)
        except:
            raise

        self._init_scalar()

    def _init_scalar(self):
        """
        By default, scalar subclasses inherit a no-op further init function.
        
        If you're deriving from ScalarBase, then you probably want to override 
        this, with the following semantics:

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Unless overridden by a derived class.

        Notes
        -----
        The derived class should take the class and construct the scalar value
        representation appropriately.
        """
        raise NotImplementedError("Help!")

    @property
    def shape(self):
        """
        Scalars are 0-dimensional and thus have no shape.
        """
        return ()

    @property
    def value(self):
        """
        The stored value of the scalar.
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = self.dtype.type(value)
        self._init_scalar()

    def as_ndarray(self):
        """
        Return a point-like ndarray containing only the value of this Scalar,
        with the dtype set accordingly.
        """
        return array(self.value, dtype=self.dtype)

    def __pow__(self, rhs):
        """
        x.__pow__(y) <==> x**y
        """
        if isinstance(rhs, ScalarBase):
            return self.result_container_type(self.value ** rhs.value,
                                              dtype = self.dtype)
        else:
            return self.result_container_type(self.value ** rhs,
                                              dtype = self.dtype)

    def __rpow__(self, rhs):
        """
        x.__rpow__(y) <==> y**x

        Notes
        -----
        For array-like types, this is computed elementwise. But ViennaCL does
        not currently support elementwise exponentiation in the scheduler, so
        this incurs the computation of the expression represented by ``y`` at
        this point. Nonetheless the result is the appropriate PyViennaCL type.
        """
        if isinstance(rhs, ScalarBase):
            return self.result_container_type(rhs ** self,
                                              dtype = self.dtype)
        else:
            return self.result_container_type(rhs ** self.value,
                                              dtype = self.dtype)
        

class HostScalar(ScalarBase):
    """
    This class is used to represent a ``host scalar``: a scalar type that is
    stored in main CPU RAM, and that is usually represented using a standard
    NumPy scalar dtype, such as int32 or float64.

    It derives from ScalarBase.
    """
    statement_node_subtype = _v.statement_node_subtype.HOST_SCALAR_TYPE
    
    def _init_scalar(self):
        self.vcl_leaf = self._value


class Scalar(ScalarBase):
    """
    This class is used to represent a ViennaCL scalar: a scalar type that is
    usually stored in OpenCL global memory, but which can be converted to a 
    HostScalar, and thence to a standard NumPy scalar dtype, such as int32 or
    float64.

    It derives from ScalarBase.
    """
    statement_node_subtype = _v.statement_node_subtype.DEVICE_SCALAR_TYPE

    def _init_scalar(self):
        try:
            vcl_type = getattr(_v, "scalar_" + vcl_statement_node_numeric_type_strings[self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("ViennaCL type %s not supported" % self.statement_node_numeric_type)
        if isinstance(self._value, vcl_type):
            self.vcl_leaf = self._value
            self._value = self._value.to_host()
        else:
            self.vcl_leaf = vcl_type(self._value)


class Vector(Leaf):
    """
    A generalised Vector class: represents ViennaCL vector objects of all
    supported scalar types. Can be constructed in a number of ways:
    * from an ndarray of the correct dtype
    * from a list
    * from an integer: produces an empty Vector of that size
    * from a tuple: first element an int (for size), second for scalar value

    Also provides convenience functions for arithmetic.
    """
    ndim = 1
    layout = None
    statement_node_type_family = _v.statement_node_type_family.VECTOR_TYPE_FAMILY
    statement_node_subtype = _v.statement_node_subtype.DENSE_VECTOR_TYPE

    def _init_leaf(self, args, kwargs):
        """
        Construct the underlying ViennaCL vector object according to the 
        given arguments and types.
        """
        
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 1:
                raise TypeError("Vector can only have a 1-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'][0])

        # TODO: Create Vector from row or column matrix..

        if len(args) == 0:
            def get_leaf(vcl_t):
                return vcl_t()
        elif len(args) == 1:
            if isinstance(args[0], MagicMethods):
                if issubclass(args[0].result_container_type, Vector):
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    def get_leaf(vcl_t):
                        return vcl_t(args[0].result.vcl_leaf)
                else:
                    raise TypeError("Vectors can only be constructed like this from one-dimensional objects")
            elif isinstance(args[0], ndarray):
                if args[0].ndim > 1:
                    one_d = [x for x in args[0].shape if x > 1]
                    if len(one_d) != 1:
                        raise TypeError("Vector can only be constructed from a one-dimensional array!")
                    a = args[0].flatten()
                else:
                    a = args[0]
                self.dtype = np_result_type(args[0])
                def get_leaf(vcl_t):
                    return vcl_t(a)
            elif isinstance(args[0], _v.vector_base):
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]
            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
        elif len(args) == 2:
            if self.dtype is None:
                try:
                    self.dtype = dtype(args[1])
                except TypeError:
                    self.dtype = np_result_type(args[1])
            def get_leaf(vcl_t):
                return vcl_t(args[0], args[1])
        else:
            raise TypeError("Vector cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = dtype(float64)

        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "vector_" + vcl_statement_node_numeric_type_strings[
                                   self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError(
                "dtype %s not supported" % self.statement_node_numeric_type)
        self.vcl_leaf = get_leaf(vcl_type)
        self.size = self.vcl_leaf.size
        self.shape = (self.size,)
        self.internal_size = self.vcl_leaf.internal_size

    def __getitem__(self, key):
        if isinstance(key, slice):
            view = View(key, self.size)
            project = getattr(_v,
                            "project_vector_" + vcl_statement_node_numeric_type_strings[
                                self.statement_node_numeric_type])
            return Vector(project(self.vcl_leaf, view.vcl_view),
                          dtype=self.dtype,
                          view_of=self,
                          view=(view,))
        elif isinstance(key, tuple) or isinstance(key, list):
            if len(key) == 0:
                return self
            elif len(key) == 1:
                return self[key[0]]
            else:
                raise IndexError("Too many indices")
        elif isinstance(key, int) or isinstance(key, long):
            # TODO: key is probably an int -- be sure?
            # TODO DUBIOUS USE INDEXERROR MAYBE?
            key = key % self.shape[0]
            return HostScalar(self.vcl_leaf.get_entry(key),
                              dtype=self.dtype)
        else:
            raise IndexError("Can't understand index")

    @property
    def index_norm_inf(self):
        """
        Returns the index of the L^inf norm on the vector.
        """
        return self.vcl_leaf.index_norm_inf

    def outer(self, rhs):
        """
        Returns the outer product of ``self`` and ``rhs``.

        Parameters
        ----------
        rhs : Vector

        Returns
        -------
        result : Matrix

        Raises
        ------
        TypeError
            If anything but a Vector is supplied.

        Notes
        -----
        ViennaCL currently does not support outer product computation in the
        expression tree, so this forces the computation of ``rhs``, if 
        ``rhs`` represents a complex expression.
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, Vector):
                return Matrix(_v.outer(self.vcl_leaf, rhs.vcl_leaf),
                              dtype=self.dtype,
                              layout=COL_MAJOR) # I don't know why COL_MAJOR..
        raise TypeError("Cannot calculate the outer-product of non-vector type: %s" % type(rhs))

    def dot(self, rhs):
        """
        Returns an expression representing the inner product of ``self`` and
        ``rhs``: ``Dot(self, rhs)``.
        """
        #return Dot(self, rhs) NOT WORKING WITH SCHEDULER
        return Scalar(_v.inner_prod(self.vcl_leaf, rhs.vcl_leaf),
                      dtype = self.dtype)
    inner = dot

    def as_column(self):
        """
        Returns a representation of this instance as a column Matrix.
        """
        tmp = self.vcl_leaf.as_ndarray()
        tmp.resize(self.size, 1)
        return Matrix(tmp, dtype=self.dtype, layout=COL_MAJOR)

    def as_row(self):
        """
        Returns a representation of this instance as a row Matrix.
        """
        tmp = self.vcl_leaf.as_ndarray()
        tmp.resize(1, self.size)
        return Matrix(tmp, dtype=self.dtype, layout=ROW_MAJOR)

    def as_diag(self):
        """
        Returns a representation of this instance as a diagonal Matrix.
        """
        tmp_v = self.as_ndarray()
        tmp_m = zeros((self.size, self.size), dtype=self.dtype)
        for i in range(self.size):
            tmp_m[i][i] = tmp_v[i]
        return Matrix(tmp_m, dtype=self.dtype) # TODO: Ought to be sparse here

    def __mul__(self, rhs):
        """
        x.__mul__(rhs) <==> x*rhs

        Returns
        -------
        z : {ElementProd(x, rhs), Mul(x, rhs)}
            Returns an ElementProd instance if rhs is a Vector, otherwise
            returns a Mul instance, which may or may not be well defined.
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, Vector):
                return ElementProd(self, rhs)
        return Mul(self, rhs)


class SparseMatrixBase(Leaf):
    """
    This is the base class for all sparse matrix types, regardless of their
    storage format.

    Because sparse matrices are leaves in the ViennaCL expression graph, this 
    class derives from the Leaf base class. It is not expected that any
    instances of this class will be created, but instead its functionality
    is provided to derived sparse matrix types.

    The specific sparse matrix subclass representing data on the compute
    device is not actually constructed until it is required; data is initially
    and transparently cached in RAM for speed of construction and access.

    A sparse matrix instance can be constructed in a number of ways:
    * as an empty instance, with no parameters;
    * by passing a 2-tuple representing the shape or a 3-tuple representing
      both the shape and the number of nonzeros, to pre-allocate memory;
    * from a ``Matrix`` instance;
    * from another sparse matrix instance;
    * from an expression resulting in a ``Matrix`` or sparse matrix;
    * from a NumPy ``ndarray``.

    Support for converting PyViennaCL sparse matrix types to and from SciPy
    sparse matrix types is not currently available, but is planned.
    """
    ndim = 2
    flushed = False
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY

    @property
    def vcl_leaf_factory(self):
        """
        Derived classes should construct and return the C++ object
        representing the appropriate sparse matrix on the compute device
        by overriding this function.
        """
        raise NotImplementedError("This is only a base class!")

    def _init_leaf(self, args, kwargs):
        """
        Do general sparse-matrix-specific construction tasks, like setting up
        the shape, layout, dtype and host memory cache.
        """
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 2:
                raise TypeError("Sparse matrix can only have a 2-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'])

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                raise TypeError("COL_MAJOR sparse layout not yet supported")
                self.layout = COL_MAJOR
            else:
                self.layout = ROW_MAJOR
        else:
            self.layout = ROW_MAJOR

        if len(args) == 0:
            # 0: empty -> empty
            def get_cpu_leaf(cpu_t):
                return cpu_t()
        elif len(args) == 1:
            if isinstance(args[0], tuple):
                if len(args[0]) == 2:
                    # 1: 2-tuple -> shape
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(args[0][0], args[0][1])
                elif len(args[0]) == 3:
                    # 1: 3-tuple -> shape+nnz
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(args[0][0], args[0][1], args[0][2])
                else:
                    # error!
                    raise TypeError("Sparse matrix cannot be constructed thus")
            elif isinstance(args[0], Matrix):
                # 1: Matrix instance -> copy
                if self.dtype is None:
                    self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0].as_ndarray())
            elif isinstance(args[0], SparseMatrixBase):
                # 1: SparseMatrixBase instance -> copy
                if self.dtype is None:
                    self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return args[0].cpu_leaf
            elif isinstance(args[0], Node):
                # 1: Node instance -> get result and copy
                result = args[0].result
                if isinstance(result, SparseMatrixBase):
                    if self.dtype is None:
                        self.dtype = result.dtype
                    self.layout = result.layout
                    def get_cpu_leaf(cpu_t):
                        return result.cpu_leaf
                elif isinstance(result, Matrix):
                    if self.dtype is None:
                        self.dtype = result.dtype
                    self.layout = result.layout
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(result.as_ndarray())
                else:
                    raise TypeError(
                        "Sparse matrix cannot be constructed thus")
            elif isinstance(args[0], ndarray):
                # 1: ndarray -> init and fill
                if self.dtype is None:
                    self.dtype = np_result_type(args[0])
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0])
            else:
                if WITH_SCIPY:
                    # then test for scipy.sparse matrix
                    raise NotImplementedError("SciPy support comes later")
                else:
                    # error!
                    raise TypeError("Sparse matrix cannot be constructed thus")
        elif len(args) == 2:
            # 2: 2 ints -> shape
            def get_cpu_leaf(cpu_t):
                return cpu_t(args[0], args[1])
        elif len(args) == 3:
            # 3: 3 ints -> shape+nnz
            def get_cpu_leaf(cpu_t):
                return cpu_t(args[0], args[1], args[2])
        else:
            raise TypeError("Sparse matrix cannot be constructed thus")

        if self.dtype is None:
            self.dtype = dtype(float64)            
        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            self.cpu_leaf_type = getattr(
                _v,
                "cpu_compressed_matrix_" + 
                vcl_statement_node_numeric_type_strings[
                    self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_numeric_type)

        self.cpu_leaf = get_cpu_leaf(self.cpu_leaf_type)
        self.base = self

    @property
    def nonzeros(self):
        """
        A ``list`` of coordinates of the nonzero elements of the matrix.
        """
        return self.cpu_leaf.nonzeros

    @property
    def nnz(self):
        """
        The number of nonzero elements stored in the matrix, as an integer.
        """
        return self.cpu_leaf.nnz

    @property
    def size1(self):
        """
        The size of the first axis of the matrix.
        """
        return self.cpu_leaf.size1

    @property
    def size2(self):
        """
        The size of the second axis of the matrix.
        """
        return self.cpu_leaf.size2

    @property
    def size(self):
        """
        The flat size (area) of the matrix, as it would be in dense form.
        """
        return self.size1 * self.size2 # Flat size

    @property
    def shape(self):
        """
        The shape of the matrix as a 2-tuple, with one entry for each axis.
        """
        return (self.size1, self.size2)

    #TODO: this doesn't work right now
    #def resize(self, size1, size2):
    #    """
    #    Resize the sparse matrix, not preserving elements.
    #    """
    #    self.flushed = False
    #    return self.cpu_leaf.resize(size1, size2)

    def as_ndarray(self):
        """
        Returns the sparse matrix as a dense NumPy ``ndarray``.
        """
        return self.cpu_leaf.as_ndarray()

    def as_dense(self):
        """
        Returns the sparse matrix as a dense PyViennaCL ``Matrix``.
        """
        return Matrix(self)

    @property
    def vcl_leaf(self):
        """
        The underlying C++ ViennaCL object representing the matrix on the
        compute device.
        """
        if not self.flushed:
            self.flush()
        return self._vcl_leaf

    def __getitem__(self, key):
        # TODO: extend beyond tuple keys
        #if not isinstance(key, tuple):
        #    raise KeyError("Key must be a 2-tuple")
        #if len(key) != 2:
        #    raise KeyError("Key must be a 2-tuple")
        #if not (isinstance(key[0], int) and isinstance(key[1], int)):
        #    raise KeyError("Only integer keys are currently supported")
        return np_result_type(self).type(self.cpu_leaf.get_entry(key[0], key[1]))

    def __setitem__(self, key, value):
        #if not isinstance(key, tuple):
        #    raise KeyError("Key must be a 2-tuple")
        #if len(key) != 2:
        #    raise KeyError("Key must be a 2-tuple")
        #if not (isinstance(key[0], int) and isinstance(key[1], int)):
        #    raise KeyError("Only integer keys are currently supported")
        self.flushed = False
        if isinstance(value, ScalarBase):
            value = value.value
        #if np_result_type(self) != np_result_type(value):
        #    value = np_result_type(self).type(value)
        self.cpu_leaf.set_entry(key[0], key[1], value)
        #self.nnz # Updates nonzero list

    def __delitem__(self, key):
        #if not isinstance(key, tuple):
        #    raise KeyError("Key must be a 2-tuple")
        #if len(key) != 2:
        #    raise KeyError("Key must be a 2-tuple")
        #if not (isinstance(key[0], int) and isinstance(key[1], int)):
        #    raise KeyError("Only integer keys are currently supported")
        self.flushed = False
        self[key] = 0
        #self.nnz # Updates nonzero list

    def __str__(self):
        out = []
        for coord in self.nonzeros:
            out += ["(", "{}".format(coord[0]), ",", "{}".format(coord[1]),
                    ")\t\t", "{}".format(self[coord]), "\n"]
        out = out[:-1]
        return "".join(out)
    __repr__ = __str__


class CompressedMatrix(SparseMatrixBase):
    """
    This class represents a sparse matrix on the ViennaCL compute device, in
    a compressed-row storage format.

    For information on construction, see the help for SparseMatrixBase.
    """
    statement_node_subtype = _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_compressed_matrix()
        self.flushed = True


class CoordinateMatrix(SparseMatrixBase):
    """
    This class represents a sparse matrix on the ViennaCL compute device, in
    a `coordinate` storage format: entries are stored as triplets 
    ``(i, j, val)``, where ``i`` is the row index, ``j`` is the column index
    and ``val`` is the entry.

    For information on construction, see the help for SparseMatrixBase.
    """
    statement_node_subtype = _v.statement_node_subtype.COORDINATE_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_coordinate_matrix()
        self.flushed = True


class ELLMatrix(SparseMatrixBase):
    """
    This class represents a sparse matrix on the ViennaCL compute device, in
    ELL storage format. In this format, the matrix is stored in a block of
    memory of size N by n_max, where N is the number of rows of the matrix
    and n_max is the maximum number of nonzeros per row. Rows with less than
    n_max entries are padded with zeros. In a second memory block, the
    respective column indices are stored.

    The ELL format is well suited for matrices where most rows have
    approximately the same number of nonzeros. This is often the case for
    matrices arising from the discretization of partial differential
    equations using e.g. the finite element method. On the other hand, the
    ELL format introduces substantial overhead if the number of nonzeros per
    row varies a lot.       [description adapted from the ViennaCL manual]

    For information on construction, see the help for SparseMatrixBase.
    """
    statement_node_subtype = _v.statement_node_subtype.ELL_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_ell_matrix()
        self.flushed = True


class HybridMatrix(SparseMatrixBase):
    """
    This class represents a sparse matrix on the ViennaCL compute device, in a
    hybrid storage format, combining the higher performance of the ELL format
    for matrices with approximately the same number of entries per row with
    the higher flexibility of the compressed row format. The main part of the 
    matrix is stored in ELL format and excess entries are stored in
    compressed row format.  [description adapted from the ViennaCL manual]

    For information on construction, see the help for SparseMatrixBase.
    """
    statement_node_subtype = _v.statement_node_subtype.HYB_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_hyb_matrix()
        self.flushed = True


# TODO: add ndarray flushing
class Matrix(Leaf):
    """
    This class represents a dense matrix object on the compute device, and it
    can be constructed in a number of ways:
    * with no parameters, as an empty matrix;
    * from an integer tuple: produces an empty Matrix of that shape;
    * from a tuple: first two values shape, third scalar value for each
      element;
    * from an ndarray of the correct dtype;
    * from a ViennaCL sparse matrix;
    * from a ViennaCL ``Matrix`` instance (to make a copy);
    * from an expression resulting in a Matrix.

    Both ROW_MAJOR and COL_MAJOR layouts are supported; to determine,
    provide ``layout`` as a keyword argument to the initialisation. The
    default layout is row-major.

    Thus, to construct a 5-by-5 column-major Matrix instance with a numeric 
    data type of ``float32`` (C++ ``float``) and each element being equal to
    ``3.141``, type:

      >>> import pyviennacl as p
      >>> mat = p.Matrix(10, 10, 3.141, dtype=p.float32, layout=p.COL_MAJOR)
      >>> print(mat)
      [[ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]]
    """
    ndim = 2
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY

    def _init_leaf(self, args, kwargs):
        """
        Construct the underlying ViennaCL vector object according to the 
        given arguments and types.
        """
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 2:
                raise TypeError("Matrix can only have a 2-d shape")
            args = list(args)
            args.insert(0, kwargs['shape'])

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                self.layout = COL_MAJOR
                self.statement_node_subtype = _v.statement_node_subtype.DENSE_COL_MATRIX_TYPE
            else:
                self.layout = ROW_MAJOR
                self.statement_node_subtype = _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE
        else:
            self.layout = ROW_MAJOR
            self.statement_node_subtype = _v.statement_node_subtype.DENSE_ROW_MATRIX_TYPE

        if len(args) == 0:
            def get_leaf(vcl_t):
                return vcl_t()
        elif len(args) == 1:
            if isinstance(args[0], MagicMethods):
                if issubclass(args[0].result_container_type,
                              SparseMatrixBase):
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    self.layout = args[0].result.layout
                    def get_leaf(vcl_t):
                        return vcl_t(args[0].result.as_ndarray())
                elif issubclass(args[0].result_container_type, Matrix):
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    self.layout = args[0].result.layout
                    def get_leaf(vcl_t):
                        return vcl_t(args[0].result.vcl_leaf)
                else:
                    raise TypeError(
                        "Matrix cannot be constructed in this way")
            elif isinstance(args[0], tuple) or isinstance(args[0], list):
                def get_leaf(vcl_t):
                    return vcl_t(args[0][0], args[0][1])
            elif isinstance(args[0], ndarray):
                if self.dtype is None:
                    self.dtype = args[0].dtype
                def get_leaf(vcl_t):
                    return vcl_t(args[0])
            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]
        elif len(args) == 2:
            if isinstance(args[0], tuple) or isinstance(args[0], list):
                if self.dtype is None:
                    self.dtype = np_result_type(args[1])
                def get_leaf(vcl_t):
                    return vcl_t(args[0][0], args[0][1], args[1])
            else:
                def get_leaf(vcl_t):
                    return vcl_t(args[0], args[1])
        elif len(args) == 3:
            if self.dtype is None:
                self.dtype = np_result_type(args[2])
            def get_leaf(vcl_t):
                return vcl_t(args[0], args[1], args[2])
        else:
            raise TypeError("Matrix cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = dtype(float64)

        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "matrix_" + 
                               vcl_layout_strings[self.layout] + "_" + 
                               vcl_statement_node_numeric_type_strings[
                                   self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_numeric_type)

        self.vcl_leaf = get_leaf(vcl_type)
        self.size1 = self.vcl_leaf.size1
        self.size2 = self.vcl_leaf.size2
        self.size = self.size1 * self.size2 # Flat size
        self.shape = (self.size1, self.size2)
        self.internal_size1 = self.vcl_leaf.internal_size1
        self.internal_size2 = self.vcl_leaf.internal_size2

    def __getitem__(self, key):
        project = getattr(_v,
                          "project_matrix_" + vcl_statement_node_numeric_type_strings[
                              self.statement_node_numeric_type])
        if isinstance(key, tuple):
            key = list(key)
        if isinstance(key, list):
            if len(key) == 0:
                return self
            elif len(key) == 1: 
                return self[key[0]]
            elif len(key) == 2:
                if isinstance(key[0], int):
                    # TODO DUBIOUS USE INDEXERROR MAYBE?
                    key[0] = key[0] % self.shape[0]
                    # Choose from row
                    if isinstance(key[1], int):
                        # TODO DUBIOUS USE INDEXERROR MAYBE?
                        key[1] = key[1] % self.shape[1]
                        #  (int, int) -> scalar
                        return HostScalar(self.vcl_leaf.get_entry(key[0], key[1]),
                                          dtype=self.dtype)
                    elif isinstance(key[1], slice):
                        #  (int, slice) - range/slice from row -> row vector
                        view1 = View(slice(key[0], key[0]+1), self.size1)
                        view2 = View(key[1], self.size2)
                        return Matrix(project(self.vcl_leaf,
                                              view1.vcl_view,
                                              view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    else:
                        raise TypeError("Did not understand key[1]")
                elif isinstance(key[0], slice):
                    # slice of rows
                    if isinstance(key[1], int):
                        # TODO DUBIOUS USE INDEXERROR MAYBE?
                        key[1] = key[1] % self.shape[1]
                        #  (slice, int) - range/slice from col -> col vector
                        view1 = View(key[0], self.size1)
                        view2 = View(slice(key[1], key[1]+1), self.size2)
                        return Matrix(project(self.vcl_leaf,
                                              view1.vcl_view,
                                              view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    elif isinstance(key[1], slice):
                        #  (slice, slice) - sub-matrix
                        view1 = View(key[0], self.size1)
                        view2 = View(key[1], self.size2)
                        return Matrix(project(self.vcl_leaf,
                                              view1.vcl_view,
                                              view2.vcl_view),
                                      dtype=self.dtype,
                                      layout=self.layout,
                                      view_of=self,
                                      view=(view1, view2))
                    else:
                        raise TypeError("Did not understand key[1]")
                else:
                    raise TypeError("Did not understand key[0]")
        elif isinstance(key, slice):
            view1 = View(key, self.size1)
            view2 = View(slice(0, self.size2, 1), self.size2)
            return Matrix(project(self.vcl_leaf,
                                  view1.vcl_view,
                                  view2.vcl_view),
                          dtype=self.dtype,
                          layout=self.layout,
                          view_of=self,
                          view=(view1, view2))
        elif isinstance(key, int):
            return self[slice(key)]
        else:
            raise IndexError("Did not understand key")

    #def clear(self):
    #    """
    #    Set every element of the matrix to 0.
    #    """
    #    return self.vcl_leaf.clear()

    @property
    def T(self):
        """
        Return the matrix transpose.
        """
        return Trans(self)
    trans = T


class Node(MagicMethods):
    """
    This is the base class for all nodes in the ViennaCL expression tree. A
    node is any binary or unary operation, such as addition. This class
    provides logic for expression tree construction and result type deduction,
    in order that expression statements can be executed correctly.

    If you're extending ViennaCL by adding an operation and want support for
    it in Python, then you should derive from this class.
    """

    statement_node_type_family = _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY
    statement_node_subtype = _v.statement_node_subtype.INVALID_SUBTYPE
    statement_node_numeric_type = _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE

    def __init__(self, *args):
        """
        Take the given operand(s) to an appropriate representation for this
        operation, and deduce the result_type. Construct a ViennaCL 
        statement_node object representing this information, ready to be
        inserted into an expression statement.
        """
        if len(args) == 1:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY
        elif len(args) == 2:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        else:
            raise TypeError("Only unary or binary nodes supported currently")

        def fix_operand(opand):
            """
            If opand is a scalar type, wrap it in a PyViennaCL scalar class.
            """
            if isinstance(opand, list):
                opand = array(opand)
            if (np_result_type(opand).name in HostScalarTypes
                and not (isinstance(opand, MagicMethods)
                         or isinstance(opand, ndarray))):
                return HostScalar(opand)
            else: return opand
        self.operands = list(map(fix_operand, args))

        if self.result_container_type is None:
            # Try swapping the operands, in case the operation supports
            # these operand types in one order but not the other; in this case
            # the mathematical intention is not ambiguous.
            self.operands.reverse()
            if self.result_container_type is None:
                # No use, so revert
                self.operands.reverse()

        self._node_init()

        if self.operation_node_type is None:
            raise TypeError("Unsupported expression: %s" % (self.express()))

        self._vcl_node_init()
        self._test_init() # Make sure we can execute

    def _node_init(self):
        pass

    def _vcl_node_init(self):
        # At the moment, ViennaCL does not do dtype promotion, so check that
        # the operands all have the same dtype.
        if len(self.operands) > 1:
            if dtype(self.operands[0]) != dtype(self.operands[1]):
                raise TypeError("dtypes on operands do not match: %s with %s and %s" % (self.express(), dtype(self.operands[0]), dtype(self.operands[1])))
            # Set up the ViennaCL statement_node with two operands
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_subtype,       # lhs
                self.operands[0].statement_node_numeric_type,  # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[1].statement_node_type_family,   # rhs
                self.operands[1].statement_node_subtype,       # rhs
                self.operands[1].statement_node_numeric_type)  # rhs
        else:
            # Set up the ViennaCL statement_node with one operand, twice..
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_subtype,       # lhs
                self.operands[0].statement_node_numeric_type,  # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                self.operands[0].statement_node_type_family,   # rhs
                self.operands[0].statement_node_subtype,       # rhs
                self.operands[0].statement_node_numeric_type)  # rhs

    def _test_init(self):
        layout_test = self.layout # NB QUIRK

    def __getitem__(self, key):
        return self.result[key]

    def __setitem__(self, key, value):
        self.result[key] = value

    def get_vcl_operand_setter(self, operand):
        """
        This function returns the correct function for setting the
        underlying ViennaCL statement_node object's operand(s) in the correct
        way: each different container type and dtype are mapped onto a
        different set_operand_to function in the underlying object, in order
        to avoid type ambiguity at the Python/C++ interface.
        """
        vcl_operand_setter = [
            "set_operand_to_",
            vcl_statement_node_subtype_strings[
                operand.statement_node_subtype],
            "_",
            vcl_statement_node_numeric_type_strings[
                operand.statement_node_numeric_type] ]
        return getattr(self.vcl_node,
                       "".join(vcl_operand_setter))

    @property
    def complexity(self):
        """
        The complexity of the ViennaCL expression, given as the number of Node
        instances in the expression tree.
        """
        complexity = 1
        for op in self.operands:
            complexity += op.complexity
        return complexity

    @property
    def result_container_type(self):
        """
        Determine the container type (ie, Scalar, Vector, etc) needed to
        store the result of the operation encoded by this Node. If the 
        operation has some effect (eg, in-place), but does not produce a
        distinct result, then return NoResult. If the operation is not
        supported for the given operand types, then return None.
        """
        if len(self.result_types) < 1:
            return NoResult

        if len(self.operands) > 0:
            try:
                op0_t = self.operands[0].result_container_type.__name__
            except AttributeError:
                # Not a PyViennaCL type, so we have a number of options.
                # Suppose an ndarray...
                if isinstance(self.operands[0], ndarray):
                    self.operands[0] = from_ndarray(self.operands[0])
                    op0_t = self.operands[0].result_container_type.__name__
                else:
                    # Otherwise, assume some scalar and hope for the best
                    op0_t = 'HostScalar'
        else:
            raise RuntimeError("What is a 0-ary operation?")

        if len(self.operands) > 1:
            try:
                op1_t = self.operands[1].result_container_type.__name__
            except AttributeError:
                if isinstance(self.operands[1], ndarray):
                    if self.operands[1].ndim == 1:
                        self.operands[1] = Vector(self.operands[1])
                    elif self.operands[1].ndim == 2:
                        self.operands[1] = Matrix(self.operands[1])
                    else:
                        raise AttributeError("Cannot cope with %d dimensions!" % self.operands[1].ndim)
                    op1_t = self.operands[1].result_container_type.__name__
                else:
                    # Again, hope for the best..
                    op1_t = 'HostScalar'
            try: return self.result_types[(op0_t, op1_t)]
            except KeyError:
                # Operation not supported for given operand types
                return None
        else:
            # No more operands, so test for 1-ary result_type
            try: return self.result_types[(op0_t, )]
            except KeyError: return None            

    @property
    def dtype(self):
        """
        Determine the dtype of the scalar element(s) of the result of the
        operation encoded by this Node, according to the NumPy type promotion
        rules.
        """
        dtypes = tuple(map(lambda x: x.dtype, self.operands))
        if len(dtypes) == 1:
            return np_result_type(dtypes[0])
        if len(dtypes) == 2:
            return np_result_type(dtypes[0], dtypes[1])

    @property
    def layout(self):
        """
        Recursively determine the storage layout for the result type, if the
        result is a Matrix.

        Notably, this ensures that any Matrix operands have the same layout,
        since this is a condition of all ViennaCL operations, except for
        the matrix-matrix product.
        """
        layout = None
        if self.result_container_type == Matrix:
            for opand in self.operands:
                try:
                    next_layout = opand.layout
                except:
                    continue
                if layout is None:
                    layout = next_layout
                if (next_layout != layout) and (self.operation_node_type != _v.operation_node_type.OPERATION_BINARY_MAT_MAT_PROD_TYPE):
                    raise TypeError("Matrices do not have the same layout")
            if layout is None:
                # May as well now choose a default layout ...
                layout = p.ROW_MAJOR
        return layout

    @property
    def result_ndim(self):
        """
        Determine the maximum number of dimensions required to store the
        result of any operation on the given operands.

        This can be overridden by the particular Node subclass, in order to
        compute the correct size for the result container.
        """
        ndim = 0
        for op in self.operands:
            if isinstance(op, Node):
                nd = op.result_ndim
                if (nd > ndim):
                    ndim = nd
            elif (op.ndim > ndim):
                ndim = op.ndim
        return ndim

    @property
    def result_max_axis_size(self):
        """
        Determine the maximum size of any axis required to store the result of
        any operation on the given operands.

        This can be overridden by the particular Node subclass, in order to
        compute the correct size for the result container.
        """
        max_size = 1
        for op in self.operands:
            if isinstance(op, Node):
                s = op.result_max_axis_size
                if (s > max_size):
                    max_size = s
            else:
                try: op.shape
                except: continue
                for s in op.shape:
                    if (s > max_size):
                        max_size = s
        return max_size

    @property
    def shape(self):
        """
        Determine the upper-bound shape of the object needed to store the
        result of any operation on the given operands. The len of this tuple
        is the number of dimensions, with each element of the tuple
        determining the upper-bound size of the corresponding dimension.

        If the shape is set manually, then this routine is overridden, and
        the manually set value is returned.
        """
        try:
            if isinstance(self._shape, tuple):
                return self._shape
        except: pass

        ndim = self.result_ndim
        max_size = self.result_max_axis_size
        shape = []
        for n in range(ndim):
            shape.append(max_size)
        shape = tuple(shape)
        self._shape = shape
        return shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value

    def express(self, statement=""):
        """
        Produce a human readable representation of the expression graph
        including all nodes and leaves connected to this one, which
        constitutes the root node.
        """
        statement += type(self).__name__ + "("
        for op in self.operands:
            statement = op.express(statement) + ", "
        if self.result_container_type is None:
            result_expression = "None"
        else:
            result_expression = self.result_container_type.__name__
        statement = statement[:-2] + ")=>" + result_expression
        return statement

    @property
    def result(self):
        """
        The result of computing the operation represented by this Node
        instance. Returns the cached result if there is one, otherwise
        executes the corresponding expression, caches the result, and returns
        that.
        """
        if not self.flushed:
            self.execute()
            return self._result
        else:
            return self._result

    @property
    def vcl_leaf(self):
        return self.result.vcl_leaf

    def execute(self):
        """
        Execute the expression tree taking this instance as the root, and
        then cache and return the result.
        """
        s = Statement(self)
        self._result = s.execute()
        self.flushed = True
        return self._result

    @property
    def value(self):
        """
        The value of the result of computing the operation represented by
        this Node; if the result is array-like, then the type is a NumPy
        ``ndarray``, otherwise, a scalar is returned.
        """
        return self.result.value

    def as_ndarray(self):
        """
        Return the value of computing the operation represented by this Node
        as a NumPy ``ndarray``.
        """
        return array(self.value, dtype=self.dtype)


class Norm_1(Node):
    """
    Represent the computation of the L^1-norm of a Vector.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_1_TYPE


class Norm_2(Node):
    """
    Represent the computation of the L^2-norm of a Vector.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_2_TYPE


class Norm_Inf(Node):
    """
    Represent the computation of the L^inf-norm of a Vector.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_INF_TYPE


class ElementAbs(Node):
    """
    Represent the elementwise computation of ``abs`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ABS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAcos(Node):
    """
    Represent the elementwise computation of ``acos`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ACOS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAsin(Node):
    """
    Represent the elementwise computation of ``asin`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ASIN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAtan(Node):
    """
    Represent the elementwise computation of ``atan`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_ATAN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCeil(Node):
    """
    Represent the elementwise computation of ``ceil`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_CEIL_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCos(Node):
    """
    Represent the elementwise computation of ``cos`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_COS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementCosh(Node):
    """
    Represent the elementwise computation of ``cosh`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_COSH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementExp(Node):
    """
    Represent the elementwise computation of ``exp`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_EXP_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementFabs(Node):
    """
    Represent the elementwise computation of ``fabs`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_FABS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementFloor(Node):
    """
    Represent the elementwise computation of ``floor`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_FLOOR_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementLog(Node):
    """
    Represent the elementwise computation of ``log`` (base e) on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_LOG_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementLog10(Node):
    """
    Represent the elementwise computation of ``log`` (base 10) on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_LOG10_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSin(Node):
    """
    Represent the elementwise computation of ``sin`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SIN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSinh(Node):
    """
    Represent the elementwise computation of ``sinh`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SINH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementSqrt(Node):
    """
    Represent the elementwise computation of ``sqrt`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_SQRT_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementTan(Node):
    """
    Represent the elementwise computation of ``tan`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TAN_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementTanh(Node):
    """
    Represent the elementwise computation of ``tanh`` on an object.
    """
    result_types = {
        ('Matrix',): Matrix,
        ('Vector',): Vector,
        ('Scalar',): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TANH_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class Trans(Node):
    """
    Represent the computation of the matrix transpose.
    """
    result_types = {
        ('Matrix',): Matrix,
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TRANS_TYPE

    def _node_init(self):
        self.shape = (self.operands[0].shape[1],
                      self.operands[0].shape[0])


class Assign(Node):
    """
    Represent the assignment (copy) of one object's content to another.
    
    For example: `x = y` is represented by `Assign(x, y)`.
    """
    result_types = {}
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE


class InplaceAdd(Assign):
    """
    Represent the computation of the in-place addition of one object to
    another.
    
    Derives from Assign rather than directly from Node because in-place
    operations are mathematically similar to assignation.
    """
    result_types = {
        #('Scalar', 'Scalar'): Scalar,
        #('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type  = _v.operation_node_type.OPERATION_BINARY_INPLACE_ADD_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot add two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape
        #if self.operands[0].result_container_type == Scalar and self.operands[1].result_container_type == HostScalar:
        #    self.operands[1] = Scalar(self.operands[1])
        #if self.operands[1].result_container_type == Scalar and self.operands[0].result_container_type == HostScalar:
        #    self.operands[0] = Scalar(self.operands[0])


class InplaceSub(Assign):
    """
    Represent the computation of the in-place subtraction of one object to
    another.
    
    Derives from Assign rather than directly from Node because in-place
    operations are mathematically similar to assignation.
    """
    result_types = {
        #('Scalar', 'Scalar'): Scalar,
        #('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type  = _v.operation_node_type.OPERATION_BINARY_INPLACE_SUB_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot subtract two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class Add(Node):
    """
    Represent the addition of one object to another.
    """
    result_types = {
        #('Scalar', 'Scalar'): Scalar,
        #('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ADD_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Add two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class Sub(Node):
    """
    Represent the subtraction of one object from another.
    """
    result_types = {
        #('Scalar', 'Scalar'): Scalar,
        #('HostScalar', 'HostScalar'): HostScalar,
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_SUB_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot Sub two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape
        if self.operands[0].result_container_type == Scalar and self.operands[1].result_container_type == HostScalar:
            temp = self.operands[1].result
            self.operands[1] = Scalar(temp)
        if self.operands[1].result_container_type == Scalar and self.operands[0].result_container_type == HostScalar:
            self.operands[0] = Scalar(self.operands[0].result)


class Mul(Node):
    """
    Represents the multiplication of one object by another.

    The semantics are as follows:
    * Scalar by scalar -> scalar;
    * scalar by vector -> scaled vector;
    * scalar by matrix -> scaled matrix;
    * vector by vector -> undefined;
    * vector by matrix -> undefined;
    * matrix by vector -> matrix-vector product;
    * matrix by matrix -> matrix-matrix product.
    
    The concern in defining these semantics has been to preserve the
    dimensionality of the operands in the result. The Mul class does not
    map directly onto the * operator for every class.
    """
    result_types = {
        # OPERATION_BINARY_MAT_MAT_PROD_TYPE
        ('Matrix', 'Matrix'): Matrix,
        # TODO: Sparse matrix support here

        # OPERATION_BINARY_MAT_VEC_PROD_TYPE
        ('Matrix', 'Vector'): Vector,
        ('CompressedMatrix', 'Vector'): Vector,
        ('CoordinateMatrix', 'Vector'): Vector,
        ('ELLMatrix', 'Vector'): Vector,
        ('HybridMatrix', 'Vector'): Vector,

        # "OPERATION_BINARY_VEC_VEC_PROD_TYPE" -- VEC as 1-D MAT?
        #('Vector', 'Vector'): Matrix, # TODO NOT IMPLEMENTED IN SCHEDULER

        # OPERATION_BINARY_MULT_TYPE
        ('Matrix', 'HostScalar'): Matrix,
        ('Matrix', 'Scalar'): Matrix,
        ('Vector', 'HostScalar'): Vector,
        ('Vector', 'Scalar'): Vector,
        ('Scalar', 'Scalar'): Scalar,
        ('Scalar', 'HostScalar'): HostScalar,
        ('HostScalar', 'HostScalar'): HostScalar
        # TODO: Sparse matrix support here
    }

    def _node_init(self):
        if (self.operands[0].result_container_type == Matrix or
            issubclass(self.operands[0].result_container_type,
                       SparseMatrixBase)): # Matrix * ...
            if (self.operands[1].result_container_type == Matrix or
                issubclass(self.operands[1].result_container_type,
                           SparseMatrixBase)):
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_MAT_PROD_TYPE
                self.shape = (self.operands[0].shape[0],
                                     self.operands[1].shape[1])
            elif self.operands[1].result_container_type == Vector:
                # Need to make sure that matrix and vector shapes are aligned
                if self.operands[0].shape[1] != self.operands[1].shape[0]:
                    raise ValueError("Operand shapes not correctly aligned")
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MAT_VEC_PROD_TYPE
                self.shape = (self.operands[0].shape[0],)
            elif self.operands[1].result_container_type == Scalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            elif self.operands[1].result_container_type == HostScalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == Vector: # Vector * ...
            if self.operands[1].result_container_type == Scalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            elif self.operands[1].result_container_type == HostScalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == Scalar: 
            #
            # TODO -- but why?..
            #
            if self.operands[1].result_container_type == Matrix:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[1].shape
            else:
                self.operation_node_type = None
        elif self.operands[0].result_container_type == HostScalar:
            #
            # TODO -- but why?..
            #
            if self.operands[1].result_container_type == Matrix:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[1].shape
            else:
                self.operation_node_type = None
        else:
            self.operation_node_type = None


class Div(Node):
    """
    Represents the division of a Matrix or Vector by a scalar.
    """
    result_types = {
        ('Vector', 'Scalar'): Vector,
        ('Vector', 'HostScalar'): Vector,
        ('Matrix', 'Scalar'): Matrix,
        ('Matrix', 'HostScalar'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_DIV_TYPE


class ElementProd(Node):
    """
    Represents the elementwise multiplication of one object by another of the
    same type.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_PROD_TYPE


class ElementDiv(Node):
    """
    Represents the elementwise multiplication of one object by another of the
    same type.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_DIV_TYPE


class Dot(Node):
    """
    Represents the computation of the inner (dot) product of two vectors.
    """
    result_types = {
        ('Vector', 'Vector'): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_INNER_PROD_TYPE
    shape = ()


class Statement:
    """
    This class represents the ViennaCL `statement` corresponding to an
    expression graph. It employs type deduction information to calculate
    the resultant types, and generates the appropriate underlying ViennaCL
    C++ object.
    """

    def __init__(self, root):
        """
        Given a Node instance, return an object representing the ViennaCL
        statement of the corresponding expression graph, as connected to the
        given root node.

        If the given root node is not an instance of Assign type, then a
        temporary object is constructed to store the result of executing the
        expression, and then a new Assign instance is created, representing
        the assignation of the result of the expression to the new temporary.
        The new Assign node is then taken to be the root node of the graph,
        having transposed the rest.
        """
        if not isinstance(root, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        self.statement = []  # A list to hold the flattened expression tree
        next_node = []       # Holds nodes as we travel down the tree

        # Test to see that we can actually do the operation
        if not root.result_container_type:
            raise TypeError("Unsupported expression: %s" %(root.express()))

        # If the root node is not an Assign instance, then construct a
        # temporary to hold the result.
        if isinstance(root, Assign):
            self.result = root.operands[0]
        else:
            self.result = root.result_container_type(
                shape = root.shape,
                dtype = root.dtype,
                layout = root.layout)
            top = Assign(self.result, root)
            next_node.append(top)

        next_node.append(root)
        # Flatten the tree
        for n in next_node:
            op_num = 0
            for operand in n.operands:
                if isinstance(operand, Node):
                    #if op_num == 0 and len(n.operands) > 1:
                    #    # ViennaCL cannot cope with complex LHS
                    #    operand = operand.result
                    #    n.operands[0] = operand
                    #    n._vcl_node_init()
                    #else:
                    next_node.append(operand)
                op_num += 1
            append_node = True
            for N in self.statement:
                if id(N) == id(n):
                    append_node = False
                    break
            if append_node:
                self.statement.append(n)

        # Contruct a ViennaCL statement object
        self.vcl_statement = _v.statement()

        # Append the nodes in the flattened statement to the ViennaCL
        # statement, doing various type checks as we go.
        for n in self.statement:
            op_num = 0
            for operand in n.operands:
                if isinstance(operand, Leaf):
                    n.get_vcl_operand_setter(operand)(op_num, operand.vcl_leaf)
                elif isinstance(operand, Node):
                    op_idx = 0
                    for next_op in self.statement:
                        if hash(operand) == hash(next_op):
                            break
                        op_idx += 1
                    n.get_vcl_operand_setter(operand)(op_num, op_idx)
                elif np_result_type(operand).name in HostScalarTypes.keys():
                    n.get_vcl_operand_setter(HostScalar(operand))(
                        op_num, operand)
                op_num += 1
            self.vcl_statement.insert_at_end(n.vcl_node)

    def execute(self):
        """
        Execute the statement -- don't do anything else -- then return the
        result (if any).
        """
        try:
            self.vcl_statement.execute()
        except RuntimeError:
            log.error("EXCEPTION EXECUTING: %s" %(self.statement[0].express()))
            raise
        return self.result


# TODO: __all__
