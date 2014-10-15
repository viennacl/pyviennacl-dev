"""Core functionality"""

from __future__ import division
import itertools, logging, math
from pyviennacl import (_viennacl as _v, backend, util)
from numpy import (ndarray, array, zeros,
                   inf, nan, dtype, number,
                   equal as np_equal, array_equal,
                   result_type as np_result_type,
                   asscalar,
                   int8, int16, int32, int64,
                   uint8, uint16, uint32, uint64,
                   float16, float32, float64)
import numpy

class NoSuchType(object):
    def __init__(self, *args, **kwargs): raise TypeError("Do not instantiate this class")

WITH_SCIPY = True
try:
    from scipy import sparse as spsparse
    scipy_sparse_type = spsparse.spmatrix
except ImportError:
    WITH_SCIPY = False
    scipy_sparse_type = NoSuchType

WITH_OPENCL = True
try:
    import pyviennacl.opencl as vcl
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    WITH_OPENCL = False

log = logging.getLogger(__name__)

# This dict maps ViennaCL container subtypes onto the strings used for them
vcl_statement_node_subtype_strings = {
    _v.statement_node_subtype.INVALID_SUBTYPE: 'node',
    _v.statement_node_subtype.HOST_SCALAR_TYPE: 'host',
    _v.statement_node_subtype.DEVICE_SCALAR_TYPE: 'scalar',
    _v.statement_node_subtype.DENSE_VECTOR_TYPE: 'vector',
    _v.statement_node_subtype.IMPLICIT_VECTOR_TYPE: 'implicit_vector',
    _v.statement_node_subtype.DENSE_MATRIX_TYPE: 'matrix',
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
vcl_statement_node_numeric_type_strings_inverse = dict([(v, k) for k, v in vcl_statement_node_numeric_type_strings.items()])

vcl_vector_base_types = {}
vcl_matrix_base_types = {}
for numeric_type in vcl_statement_node_numeric_type_strings.values():
    try: vcl_vector_base_types[numeric_type] = getattr(_v, 'vector_base_' + numeric_type)
    except: pass
    try: vcl_matrix_base_types[numeric_type] = getattr(_v, 'matrix_base_' + numeric_type)
    except: pass

mem_handle_types = [backend.MemoryHandle]
if WITH_OPENCL:
    mem_handle_types.append(cl.MemoryObject)
    mem_handle_types.append(cl.array.Array)

# This dict is used to map NumPy dtypes onto OpenCL/ViennaCL numeric types
HostScalarTypes = {
    'int8': _v.statement_node_numeric_type.CHAR_TYPE,
    'int16': _v.statement_node_numeric_type.SHORT_TYPE,
    'int32': _v.statement_node_numeric_type.INT_TYPE,
    'int64': _v.statement_node_numeric_type.LONG_TYPE,
    'int': _v.statement_node_numeric_type.LONG_TYPE,
    'uint8': _v.statement_node_numeric_type.UCHAR_TYPE,
    'uint16': _v.statement_node_numeric_type.USHORT_TYPE,
    'uint32': _v.statement_node_numeric_type.UINT_TYPE,
    'uint64': _v.statement_node_numeric_type.ULONG_TYPE,
    'float16': _v.statement_node_numeric_type.HALF_TYPE,
    'float32': _v.statement_node_numeric_type.FLOAT_TYPE,
    'float64': _v.statement_node_numeric_type.DOUBLE_TYPE,
    'float': _v.statement_node_numeric_type.DOUBLE_TYPE
}
HostScalarTypes_inverse = dict([(v, k) for k, v in HostScalarTypes.items()])

# Constants for choosing matrix storage layout
ROW_MAJOR = 'C'
COL_MAJOR = 'F'

vcl_layout_strings = {
    ROW_MAJOR: 'row',
    COL_MAJOR: 'col'
}


class NoResult(object): 
    """This no-op class is used to represent when some ViennaCL operation
    produces no explicit result, aside from any effects it may have on
    the operands.
    
    For instance, in-place operations can return :class:`NoResult`, as can
    :class:`Assign`.
    """
    pass

def noop(*args):
    """A no-op function.
    """
    pass


class MagicMethods(object):
    """A class to provide convenience methods for arithmetic and BLAS
    access.

    Classes derived from this will inherit lots of useful features
    applicable to PyViennaCL. For more information, see the individual
    methods below.
    """
    flushed = False
    no_fix = False

    @property
    def itemsize(self):
        """Return the size in bytes of an element of this object.
        """
        return np_result_type(self).itemsize

    def as_opencl_array(self):
        """Return a representation of this object as a PyOpenCL :class:`Array`.
        """
        if self.context.domain is not backend.OpenCLMemory:
            raise TypeError("This operation is currently only supported with the OpenCL backend")
        c = cl.array.Array(self.context.current_queue,
                           self.shape,
                           self.dtype,
                           order = self.layout,
                           data = self.handle[0].buffer,
                           strides = self.strides)
        return c

    def as_opencl_kernel_operands(self):
        raise NotImplementedError("This needs to be overridden by derived classes")

    def result_container_type(self):
        """This function should be overridden, with the following semantics.

        :returns: *type*

           The type that the operation or object represented by an instance
           of this class should return as a result on execution.

        :raises: *NotImplementedError*

           If you do not override this function in a class derived from
           :class:`MagicMethods`.
        """
        raise NotImplementedError("Why is this happening to you?")

    def copy(self):
        """Returns a new instance of this class representing a new copy of
        this instance's data.
        """
        return self.result_container_type(self)

    def new_instance(self, data=None):
        """Returns a new instance of this class. By default, the new instance
        will be empty, inheriting none of the data of the current
        instance.
        """
        if data is None:
            return self.result_container_type(shape = self.shape,
                                              dtype = self.dtype,
                                              layout = self.layout,
                                              context = self.context)
        else:
            return self.result_container_type(data, shape = self.shape,
                                              dtype = self.dtype,
                                              layout = self.layout,
                                              context = self.context)

    def norm(self, ord=None):
        """Returns a norm of this instance, if that is defined.
        
        The norm returned depends on the *ord* parameter, as in SciPy.

        :param: *ord* -- {1, 2, inf, 'fro', None}

           Order of the norm.

           *inf* means NumPy's :class:`inf` object.

           *'fro'* means the string 'fro', and denotes the Frobenius norm.

           If None and ``self`` is a Matrix instance, then assumes 'fro'.
        """
        if ord is None: # TODO: Tidy this up
            try:
                return self.norm('fro')
            except:
                return self.norm(2)

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

    def element_prod(self, rhs):
        """Returns the elementwise product of *self* and *rhs*, for some *rhs*.
        """
        return ElementProd(self, rhs)
    element_mul = element_prod

    def element_div(self, rhs):
        """Returns the elementwise division of *self* and *rhs*, for some *rhs*
        """
        return ElementDiv(self, rhs)

    def __pow__(self, rhs):
        """x.__pow__(y) <==> x**y

        .. note ::

           For array-like types, this is computed
           elementwise. But ViennaCL does not currently support
           elementwise exponentiation in the scheduler, so this incurs
           the computation of the expression represented by *y* at this
           point. Nonetheless, the result is the appropriate PyViennaCL
           type.

        """
        return ElementPow(self, rhs)

    def __eq__(self, rhs):
        """The equality operator.

        :param: *rhs* -- {scalar, Vector, Matrix, ndarray, etc}

        :returns: {*bool*, *ndarray*}

           If the r.h.s. is elementwise comparable with a
           :class:`Vector`, :class:`Matrix` or :class:`ndarray`, then
           an array of boolean values is returned; see NumPy's *equal*
           function. If the r.h.s. is a scalar, then a boolean value
           is returned. Otherwise, the behaviour is undefined, but the
           Python ``==`` operator is used to compare the *result*
           attribute of *self* to the r.h.s., in an attempt at
           meaningfulness.

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

    def __lt__(self, rhs):
        """The less-than operator.

        :param: *rhs* -- {scalar, Vector, Matrix, ndarray, etc}

        :returns: {*bool*, *ndarray*}

           If the r.h.s. is elementwise comparable with a
           :class:`Vector`, :class:`Matrix` or :class:`ndarray`, then
           an array of boolean values is returned. If the r.h.s. is a
           scalar, then a boolean value is returned.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.value < rhs.value
            else:
                return self.value < rhs
        if self.flushed:
            if isinstance(rhs, MagicMethods):
                return self.as_ndarray() < rhs.as_ndarray()
            elif isinstance(rhs, ndarray):
                return self.as_ndarray() < rhs
            else:
                return self.value < rhs
        else:
            return self.result < rhs

    def __gt__(self, rhs):
        """The greater-than operator.

        :param: *rhs* -- {scalar, Vector, Matrix, ndarray, etc}

        :returns: {*bool*, *ndarray*}

           If the r.h.s. is elementwise comparable with a
           :class:`Vector`, :class:`Matrix` or :class:`ndarray`, then
           an array of boolean values is returned. If the r.h.s. is a
           scalar, then a boolean value is returned.
        """
        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.value > rhs.value
            else:
                return self.value > rhs
        if self.flushed:
            if isinstance(rhs, MagicMethods):
                return self.as_ndarray() > rhs.as_ndarray()
            elif isinstance(rhs, ndarray):
                return self.as_ndarray() > rhs
            else:
                return self.value > rhs
        else:
            return self.result > rhs

    def __le__(self, rhs):
        return not (self > rhs)

    def __ge__(self, rhs):
        return not (self < rhs)

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
        """x.__add__(y) <==> x+y
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
        """x.__sub__(y) <==> x-y
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
        """x.__mul__(y) <==> x*y

        :returns: *z* : {Mul(x, y), (x.value * rhs)}

           Returns a :class:`Mul` instance if defined.
        """
        op = Mul(self, rhs)
        if op.result_container_type is None:
            op = ElementProd(self, rhs)
        if op.result_container_type is not None:
            return op

        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value * rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.result_container_type(self.value * rhs,
                                                  dtype = self.dtype)
        else:
            raise TypeError("I don't know how we got into this situation: can't express the multiplication!")

    def __matmul__(self, rhs):
        """x.__matmul__(y) <==> x@y

        :returns: *z* : {Mul(x, y), Dot(x, y)}

           Returns a :class:`Dot` instance for two vectors, or a
           :class:`Mul` instance.
        """
        op = Dot(self, rhs)
        if op.result_container_type is None:
            op = Mul(self, rhs)
        return op
    dot = __matmul__

    def __floordiv__(self, rhs):
        """x.__floordiv__(y) <==> x//y
        """
        op = math.floor(self.__truediv__(rhs))
        return op

    def __truediv__(self, rhs):
        """x.__truediv__(y) <==> x/y

        .. note ::

           PyViennaCL automatically adopts Python 3.0 division semantics, so
           the ``/`` division operator is never floor (integer) division, and
           always true floating point division.
        """
        op = Div(self, rhs)
        if op.result_container_type is None:
            op = ElementDiv(self, rhs)
        if op.result_container_type is not None:
            return op

        if issubclass(self.result_container_type, ScalarBase):
            if isinstance(rhs, MagicMethods):
                if issubclass(rhs.result_container_type, ScalarBase):
                    return self.result_container_type(self.value / rhs.value,
                                                      dtype = self.dtype)
            else:
                return self.result_container_type(self.value / rhs,
                                                  dtype = self.dtype)
        else:
            raise TypeError("I don't know how we got into this situation: can't express the division!")
    __div__ = __truediv__

    def __iadd__(self, rhs):
        """x.__iadd__(y) <==> x+=y
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
        """x.__isub__(y) <==> x-=y
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
        """x.__imul__(y) <==> x*=y
       
        .. note ::

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
        """x.__ifloordiv__(y) <==> x//=y
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
        """x.__itruediv__(y) <==> x/=y

        .. note ::

           PyViennaCL automatically adopts Python 3.0 division semantics, so
           the ``/`` division operator is never floor (integer) division, and
           always true floating point division.
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
        """x.__radd__(y) <==> y+x

        :note: Addition is commutative.
        """
        return self + rhs

    def __rsub__(self, rhs):
        """x.__rsub__(y) <==> y-x
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value - self.value,
                                                 dtype = rhs.dtype)
            return rhs - self
        return rhs - self.value

    def __rmul__(self, rhs):
        """x.__rmul__(y) <==> y*x

        :note: Multiplication is commutative.
        """
        return self * rhs

    def __rfloordiv__(self, rhs):
        """x.__rfloordiv__(y) <==> y//x
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value // self.value,
                                                 dtype = rhs.dtype)
            return rhs // self
        return rhs // self.value

    def __rtruediv__(self, rhs):
        """x.__rtruediv__(y) <==> y/x

        .. note ::

           PyViennaCL automatically adopts Python 3.0 division semantics, so
           the ``/`` division operator is never floor (integer) division, and
           always true floating point division.
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return rhs.result_container_type(rhs.value / self.value,
                                                 dtype = rhs.dtype)
            return rhs / self
        return rhs / self.value
    __rdiv__ = __rtruediv__

    def __neg__(self):
        """x.__neg__() <==> -x
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(-self.value, dtype = self.dtype)
        return Neg(self)

    def __abs__(self):
        """x.__abs__() <==> abs(x)

        .. note ::

           OpenCL does not provide for ``abs`` on floating point types, so if
           your instance has a floating point data type, ``abs(x)`` is
           equivalent to ``fabs(x)``.

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
        """x.__floor__() <==> math.floor(self)

        .. note ::

           On array-like types, this is computed elementwise.
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(math.floor(self.value),
                                              dtype = self.dtype)
        return ElementFloor(self)

    def __ceil__(self):
        """x.__ceil__() <==> math.ceil(self)
        
        .. note ::

           On array-like types, this is computed elementwise.
        """
        if issubclass(self.result_container_type, ScalarBase):
            return self.result_container_type(math.ceil(self.value),
                                              dtype = self.dtype)
        return ElementCeil(self)

    def diag(self, k=0, layout=ROW_MAJOR):
        """TODO docstring
        """
        if issubclass(self.result_container_type, Matrix):
            return Vector(_v.diag(self.vcl_leaf, k), context=self.context)
        elif issubclass(self.result_container_type, Vector):
            if layout == ROW_MAJOR:
                return Matrix(_v.diag_row(self.vcl_leaf, k), layout=layout, context=self.context)
            else:
                return Matrix(_v.diag_col(self.vcl_leaf, k), layout=layout, context=self.context)
        else:
            raise TypeError("Can only get a diagonal from a Matrix or Vector!")

    def row(self, index):
        """If *self* is of :class:`Matrix` type, then return the row at
        *index* as a :class:`Vector`. Otherwise, raise :exc:`TypeError`.

        .. note ::

           This will cause the computation of any expression
           represented by *self*.
        """
        if not issubclass(self.result_container_type, Matrix):
            raise TypeError("Can only get rows from a dense Matrix!")

        return Vector(_v.row(self.vcl_leaf, index), context=self.context)

    def column(self, index):
        """If *self* is of :class:`Matrix` type, then return the column at
        *index* as a :class:`Vector`. Otherwise, raise :exc:`TypeError`.

        .. note ::

           This will cause the computation of any expression
           represented by *self*.
        """
        if not issubclass(self.result_container_type, Matrix):
            raise TypeError("Can only get columns from a dense Matrix!")

        return Vector(_v.column(self.vcl_leaf, index), context=self.context)

    @property
    def index_norm_inf(self):
        """Returns the index of the L^inf norm on the vector.
        TODO docstring (now in MagicMethods)
        """
        if not issubclass(self.result_container_type, Vector):
            raise TypeError("Operation not defined on non-Vector types")

        return self.vcl_leaf.index_norm_inf

    def outer(self, rhs, layout=ROW_MAJOR):
        """Returns the outer product of *self* and *rhs*.

        :param: :class:`Vector`.

        :returns: :class:`Matrix` with given layout or ROW_MAJOR

        .. note ::

           ViennaCL currently does not support outer product computation in
           the expression tree, so this forces the computation of *rhs*, if 
           *rhs* represents a complex expression.
        TODO docstring (now in MagicMethods)
        """
        if not issubclass(self.result_container_type, Vector):
            raise TypeError("Operation not defined on non-Vector types")

        #return self.as_column(layout=layout).dot(rhs.as_row(layout=layout))
        if layout == ROW_MAJOR:
            return Matrix(_v.outer_row(self.vcl_leaf, rhs.vcl_leaf),
                          dtype=self.dtype,
                          layout=ROW_MAJOR,
                          context=self.context)
        else:
            return Matrix(_v.outer_col(self.vcl_leaf, rhs.vcl_leaf),
                          dtype=self.dtype,
                          layout=COL_MAJOR,
                          context=self.context)

    def as_column(self, layout=ROW_MAJOR, copy=False):
        """Returns a representation of this instance as a column
        :class:`Matrix`.
        TODO docstring (now in MagicMethods)
        """
        if not issubclass(self.result_container_type, Vector):
            raise TypeError("Operation not defined on non-Vector types")

        if copy: tmp = self.copy()
        else: tmp = self
        return Matrix(tmp.handle[0], dtype=tmp.dtype, layout=layout,
                      shape=(tmp.shape[0], 1),
                      internal_shape=(tmp.internal_shape[0], 1),
                      offset=(tmp.offset, 0),
                      strides=(tmp.strides[0], tmp.strides[0] * tmp.internal_shape[0]),
                      context=self.context)

    def as_row(self, layout=ROW_MAJOR, copy=False):
        """Returns a representation of this instance as a row
        :class:`Matrix`.
        TODO docstring (now in MagicMethods)
        """
        if not issubclass(self.result_container_type, Vector):
            raise TypeError("Operation not defined on non-Vector types")

        if copy: tmp = self.copy()
        else: tmp = self
        return Matrix(tmp.handle[0], dtype=tmp.dtype, layout=layout,
                      shape=(1, tmp.shape[0]),
                      internal_shape=(1, tmp.internal_shape[0]),
                      offset=(0, tmp.offset),
                      strides=(tmp.strides[0] * tmp.internal_shape[0], tmp.strides[0]),
                      context=self.context)

    def as_diag(self):
        """Returns a representation of this instance as a diagonal
        :class:`Matrix`.
        TODO docstring (now in MagicMethods)
        """
        if not issubclass(self.result_container_type, Vector):
            raise TypeError("Operation not defined on non-Vector types")

        tmp_v = self.as_ndarray()
        tmp_m = zeros((self.size, self.size), dtype=self.dtype)
        for i in range(self.size):
            tmp_m[i][i] = tmp_v[i]
        return Matrix(tmp_m, dtype=self.dtype,
                      layout=self.layout, context=self.context) # TODO: Ought to be sparse here


class View(object):
    """This class represents a C++ ViennaCL range or slice, as a 'view'
    on an object. A *View* instance is object-independent; i.e., it
    represents an abstract view.
    """
    start = None
    stop = None
    step = None

    def __init__(self, key, axis_size):
        """Construct a View object.

        :param: *key* : slice

        :param: *axis_size* : int

           The number of elements along the axis of which the instance
           of this class is a view.
        """
        start, stop, step = key.indices(axis_size)
        if stop < start: stop += axis_size
        size = stop - start

        if step == 1:
            # then range -- or slice!
            self.vcl_view = _v.slice(start, 1, size)
        else:
            # then slice
            self.vcl_view = _v.slice(start, step,
                                     int(math.ceil(size/step)))

        self.slice = key
        self.start = start
        self.stop = stop
        self.step = step


class Leaf(MagicMethods):
    """This is the base class for all leaves in the ViennaCL expression tree
    system. A leaf is any type that can store data for use in an operation,
    such as a scalar, a vector, or a matrix.

    You can provide the following to the constructor, either as arguments or
    keyword arguments:

    :param: *context*

       Context within which to create the Vector. Can be one of
       :class:`backend.Context`, :class:`backend.MemoryDomain`, or
       :class:`pyopencl.Context`.

    :param: *dtype*

       Numerical data type of each element of the Vector.
    """
    shape = None   # No shape yet -- not even 0 dimensions
    flushed = True # Are host and device data synchronised?
    complexity = 0 # A leaf does not contribute computationally to a tree

    def __init__(self, *args, **kwargs):
        """Do initialisation tasks common to all Leaf subclasses, then pass
        control onto the overridden ``_init_leaf`` function.

        Tasks include expression computation and configuration of data types
        and views.
        """
        self._context = None
        self.dtype = None

        args = list(args)
        for arg in args:
            REMOVE_ARG = False

            if isinstance(arg, list):
                for item in arg:
                    if isinstance(item, MagicMethods):
                        arg[arg.index(item)] = item.value

            #if isinstance(arg, number):
            #    args[args.index(arg)] = asscalar(arg)

            ARG_IS_NUMBER = False
            try:
                if issubclass(arg, number) or issubclass(arg, dtype):
                    ARG_IS_NUMBER = True
            except TypeError: pass
            if ARG_IS_NUMBER:
                self.dtype = np_result_type(arg)
                REMOVE_ARG = True

            ARG_IS_MEM_DOMAIN = False
            try:
                if issubclass(arg, backend.MemoryDomain):
                    ARG_IS_MEM_DOMAIN = True
            except TypeError: pass
            if ARG_IS_MEM_DOMAIN or isinstance(arg, backend.Context):
                self._context = backend.Context(arg)
                REMOVE_ARG = True
            elif WITH_OPENCL:
                if isinstance(arg, cl.Context):
                    self._context = backend.Context(arg)
                    REMOVE_ARG = True

            if REMOVE_ARG:
                args.remove(arg)

        if 'context' in kwargs.keys():
            self._context = backend.Context(kwargs['context'])
        elif self._context is None:
            self._context = backend.default_context

        if 'dtype' in kwargs.keys():    
            self.dtype = dtype(kwargs['dtype']) 

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
                        log.error("Failed to convert value dtype (%s) to item dtype (%s)" % 
                                      (np_result_type(item), np_result_type(value)))
                try:
                    try: # Assume matrix type
                        return self.vcl_leaf.set_entry(key[0], key[1], value) ## set
                    except: # Otherwise, assume vector
                        return self.vcl_leaf.set_entry(key, value) ## set
                except:
                    log.error("Failed to set vcl entry")
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
        """By default, leaf subclasses inherit a no-op further init function.
        
        If you're deriving from :class:`Leaf`, then you probably want
        to override this, with the following semantics:

        :raises: *NotImplementedError*

           Unless overridden by a derived class.

        .. note ::

           The derived class should take *args* and *kwargs* and construct
           the leaf appropriately.
        """
        raise NotImplementedError("Help")

    def flush(self):
        """Override this function to implement caching functionality.
        """
        raise NotImplementedError("Should you be trying to flush this type?")

    @property
    def handle(self):
        # TODO: Need setter
        return self._handle

    @property
    def context(self):
        # TODO: Need setter
        return self._context

    @property
    def result_container_type(self):
        """The result_container_type for a leaf is always its own type.
        """
        return type(self)

    @property
    def result(self):
        """The result of an expression or subexpression consisting of a leaf
        is just the leaf itself.
        """
        return self

    def express(self, statement=""):
        """Construct a human-readable version of a ViennaCL expression tree
        statement from this leaf.
        """
        statement += type(self).__name__ + ":" + str(dtype(self))
        return statement

    def as_ndarray(self):
        """Return a NumPy :class:`ndarray` containing the data within the
        underlying ViennaCL type.
        """
        return array(self.vcl_leaf.as_ndarray(), dtype=self.dtype)

    @property
    def value(self):
        """Return a NumPy :class:`ndarray` containing the data within the
        underlying ViennaCL type.
        """
        return self.as_ndarray()


class ScalarBase(Leaf):
    """This is the base class for all scalar types, regardless of their
    memory and backend context. It represents the dtype and the value
    of the scalar independently of one another.

    Because scalars are leaves in the ViennaCL expression graph, this class
    derives from the :class:`Leaf` base class.
    """
    statement_node_type_family = _v.statement_node_type_family.SCALAR_TYPE_FAMILY
    ndim = 0 # Scalars are point-like, and thus 0-dimensional

    def _init_leaf(self, args, kwargs):
        """Do Scalar-specific initialisation tasks.
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

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = np_result_type(self._value) #self._context.default_dtype

        try:
            self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]
        except KeyError:
            raise TypeError("dtype %s not supported" % self.dtype.name)
        except:
            raise

        self._init_scalar()

    def _init_scalar(self):
        """By default, scalar subclasses inherit a no-op init function.
        
        If you're deriving from ScalarBase, then you want to override this,
        with the following semantics:

        :raises: *NotImplementedError*

           Unless overridden by a derived class.

        .. note ::

           The derived class should take the class and construct the scalar
           value representation appropriately.
        """
        raise NotImplementedError("Help!")

    @property
    def shape(self):
        """Scalars are 0-dimensional and thus have no shape.
        """
        return ()

    @property
    def value(self):
        """The stored value of the scalar.
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = self.dtype.type(value)
        self._init_scalar()

    def as_ndarray(self):
        """Return a point-like ndarray containing only the value of this
        Scalar, with the dtype set accordingly.
        """
        return array(self.value, dtype=self.dtype)

    def as_opencl_kernel_operands(self):
        """Returns a representation of the current object sufficient for
        passing to PyOpenCL for executing a custom kernel.
        """
        return [np_result_type(self).type(self.value)]

    def __pow__(self, rhs):
        """x.__pow__(y) <==> x**y
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return self.result_container_type(self.value ** rhs.value,
                                                  dtype = self.dtype)
        else:
            return self.result_container_type(self.value ** rhs,
                                              dtype = self.dtype)

    def __rpow__(self, rhs):
        """x.__rpow__(y) <==> y**x

        .. note ::

           For array-like types, this is computed elementwise. But ViennaCL
           does not currently support elementwise exponentiation in the
           scheduler, so this incurs the computation of the expression
           represented by *y* at this point. Nonetheless, the result is the
           appropriate PyViennaCL type.
        """
        if isinstance(rhs, MagicMethods):
            if issubclass(rhs.result_container_type, ScalarBase):
                return self.result_container_type(rhs.value ** self.value,
                                                  dtype = self.dtype)
        else:
            return self.result_container_type(rhs ** self.value,
                                              dtype = self.dtype)


class HostScalar(ScalarBase):
    """This class is used to represent a *host scalar*: a scalar type
    that is stored in main CPU RAM, and that is usually represented
    using a standard NumPy scalar dtype, such as int32 or float64.

    Construct an instance from a numerical value, using a built-in Python type,
    such as :class:`float`, or a NumPy type, such as :class:`float32`, or
    another PyViennaCL :class:`ScalarBase` instance.

    See also the constructor parameters inherited from :class:`Leaf`.

    Also inherits convenience functions for arithmetic; see
    :class:`MagicMethods`.
    """
    statement_node_subtype = _v.statement_node_subtype.HOST_SCALAR_TYPE
    
    def _init_scalar(self):
        self.vcl_leaf = self._value
        self._handle = (None,)
        #self._context = None


class Scalar(ScalarBase):
    """This class is used to represent a ViennaCL scalar: a scalar type
    that is usually stored in OpenCL global memory, but which can be
    converted to a :class:`HostScalar`, and thence to a standard NumPy
    scalar dtype, such as :class:`int32` or :class:`float64`.

    Construct an instance from a numerical value, using a built-in Python type,
    such as :class:`float`, or a NumPy type, such as :class:`float32`, or
    another PyViennaCL :class:`ScalarBase` instance.

    See also the constructor parameters inherited from :class:`Leaf`.

    Also inherits convenience functions for arithmetic; see
    :class:`MagicMethods`.
    """
    statement_node_subtype = _v.statement_node_subtype.DEVICE_SCALAR_TYPE

    def _init_scalar(self):
        try:
            vcl_type = getattr(_v, "scalar_" + vcl_statement_node_numeric_type_strings[self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("ViennaCL type %s not supported" % self.statement_node_numeric_type)

        if isinstance(self._value, vcl_type):
            self._value = self._value.to_host()
        #if isinstance(self._value, number):
        #    self._value = asscalar(self._value)
        self.vcl_leaf = vcl_type(self._value, self._context.vcl_context)
        self._handle = (backend.MemoryHandle(self.vcl_leaf.handle),)

    @property
    def value(self):
        """The stored value of the scalar.
        """
        return self.vcl_leaf.to_host()

    @value.setter
    def value(self, value):
        self._value = self.dtype.type(value)
        self._init_scalar()


class Vector(Leaf):
    """A generalised Vector class: represents ViennaCL vector objects of all
    supported scalar types. Can be constructed in a number of ways:

    * from an ndarray of the correct dtype;
    * from a list;
    * from an integer: produces an empty Vector of that size;
    * from a tuple: first element an int (for size), second for scalar value;
    * from a PyOpenCL *Array*: producing a view onto the associated buffer;
    * from a host / OpenCL / CUDA buffer, wrapped in a
      :class:`backend.MemoryHandle` object.

    You can also provide the following keyword parameters to the constructor:

    :param: *size*

       Size of the vector to construct.

    :param: *shape* = (size,)

       A tuple of one element giving the size of the Vector.

       You should only pass one of *shape* or *size*.

    And, if you are constructing fom a buffer via a
    :class:`backend.MemoryHandle` object:

    :param: *offset* : int

       The starting index for the elements of the vector in the buffer, given
       as a multiple of the size in bytes of each element. The default is 0.

    :param: *stride* : int

       The number of bytes between elements of the Vector in the buffer, as a
       multiple of the size of each element. The default is 1.

    See also the constructor parameters inherited from :class:`Leaf`.

    Also inherits convenience functions for arithmetic; see
    :class:`MagicMethods`.

    """
    ndim = 1
    layout = None
    statement_node_type_family = _v.statement_node_type_family.VECTOR_TYPE_FAMILY
    statement_node_subtype = _v.statement_node_subtype.DENSE_VECTOR_TYPE

    def _init_leaf(self, args, kwargs):
        """Construct the underlying ViennaCL vector object according to the
        given arguments and types.
        """
        
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 1:
                raise TypeError("Vector can only have a 1-d shape")
            shape = kwargs['shape']
        elif 'size' in kwargs.keys():
            shape = (kwargs['size'],)
        else:
            shape = ()

        # TODO: Create Vector from row or column matrix..

        if len(args) == 0:
            if shape:
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], self._context.vcl_context)
            else:
                def get_leaf(vcl_t):
                    return vcl_t(self._context.vcl_context)
                
        elif len(args) == 1:
            if isinstance(args[0], MagicMethods):
                if issubclass(args[0].result_container_type, Vector):
                    if args[0].handle[0].domain is not self._context.domain:
                        raise TypeError("Can only construct from objects with same memory domain")
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    if not shape:
                        shape = args[0].shape
                    elif shape != args[0].shape:
                        raise TypeError("Shapes not compatible")
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
                if not shape:
                    shape = args[0].shape
                elif shape != args[0].shape:
                    raise TypeError("Shapes not compatible")
                def get_leaf(vcl_t):
                    return vcl_t(a, self._context.vcl_context)

            elif isinstance(args[0], tuple(vcl_vector_base_types.values())):
                # Crudely find the dtype of the argument
                test_dtype = None
                for key in vcl_vector_base_types.keys():
                    if isinstance(args[0], vcl_vector_base_types[key]):
                        test_dtype = dtype(getattr(numpy, HostScalarTypes_inverse[vcl_statement_node_numeric_type_strings_inverse[key]]))
                        break

                if test_dtype is None:
                    raise TypeError("Could not deduce dtype of argument")
                if self.dtype is None:
                    self.dtype = test_dtype
                elif test_dtype != self.dtype:
                    raise TypeError("Cannot convert dtypes")

                if backend.vcl_memory_types[args[0].memory_domain] is not self._context.domain:
                    raise TypeError("Can only construct from objects with same memory domain")

                def get_leaf(vcl_t):
                    return args[0]

            elif isinstance(args[0], tuple(mem_handle_types)):
                mem_handle = args[0]
                WITH_CL_ARRAY = False
                if WITH_OPENCL:
                    if isinstance(mem_handle, cl.MemoryObject):
                        mem_handle = backend.MemoryHandle(mem_handle)
                    elif isinstance(mem_handle, cl.array.Array):
                        cl_array = mem_handle
                        if len(cl_array.shape) > 1:
                            raise TypeError("Can only construct a Vector from a 1-D array!")
                        mem_handle = backend.MemoryHandle(cl_array.base_data)
                        WITH_CL_ARRAY = True

                if self.dtype is None:
                    if WITH_CL_ARRAY:
                        self.dtype = np_result_type(cl_array)
                    else:
                        raise TypeError("You must set the dtype if constructing from a MemoryHandle")

                if mem_handle.context is None:
                    mem_handle.context = self._context

                if shape:
                    size = shape[0]
                else:
                    if WITH_CL_ARRAY:
                        size = cl_array.size
                    else:
                        size = int(mem_handle.raw_size / self.itemsize)

                if 'offset' in kwargs.keys():
                    try: offset = int(kwargs['offset'][0] / self.itemsize)
                    except TypeError: offset = int(kwargs['offset'] / self.itemsize)
                elif WITH_CL_ARRAY:
                    offset = int(cl_array.offset / self.itemsize)
                else:
                    offset = 0

                if 'strides' in kwargs.keys():
                    stride = int(kwargs['strides'][0] / self.itemsize)
                elif 'stride' in kwargs.keys():
                    stride = int(kwargs['stride'] / self.itemsize)
                elif WITH_CL_ARRAY:
                    stride = int(cl_array.strides[0] / self.itemsize)
                else:
                    stride = 1

                def get_leaf(vcl_t):
                    base_t = vcl_t.__bases__[0]
                    return base_t(mem_handle.vcl_handle, size, offset, stride)

            else:
                # This doesn't do any shape or dtype checking, so beware...
                def get_leaf(vcl_t):
                    try:
                        return vcl_t(args[0], self._context.vcl_context)
                    except:
                        return vcl_t(args[0])

        elif len(args) == 2:
            if self.dtype is None:
                self.dtype = np_result_type(args[1])
            if not shape:
                def get_leaf(vcl_t):
                    return vcl_t(args[0], args[1], self._context.vcl_context)
            else:
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], args[1], self._context.vcl_context)

        else:
            raise TypeError("Vector cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = self._context.default_dtype

        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            vcl_type = getattr(_v,
                               "vector_" + vcl_statement_node_numeric_type_strings[
                                   self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError(
                "dtype %s not supported" % self.statement_node_numeric_type)

        self.vcl_leaf = get_leaf(vcl_type)
        self._handle = (backend.MemoryHandle(self.vcl_leaf.handle),)
        self.layout = ROW_MAJOR

    @property
    def size(self):
        return self.vcl_leaf.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def internal_size(self):
        return self.vcl_leaf.internal_size

    @property
    def internal_shape(self):
        return (self.internal_size,)

    @property
    def strides(self):
        return (self.vcl_leaf.stride * self.itemsize,)

    @property
    def offset(self):
        return self.vcl_leaf.start * self.strides[0]

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
            if abs(key) >= self.size:
                raise IndexError("Index beyond end of vector")
            if key < 0:
                key += self.size
            return HostScalar(self.vcl_leaf.get_entry(key),
                              dtype=self.dtype)
        else:
            raise IndexError("Can't understand index")

    def as_opencl_kernel_operands(self):
        """Returns a representation of the current object sufficient for
        passing to PyOpenCL for executing a custom kernel.

        In this case, this means a list consisting of the buffer for this
        object and its size.
        """
        return [self.handle[0].buffer, uint32(self.internal_size)]


class SparseMatrixBase(Leaf):
    """This is the base class for all sparse matrix types, regardless of
    their storage format.

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
    * from a 3-tuple of lists representing (rows, cols, values);
    * from a :class:`Matrix` instance;
    * from another sparse matrix instance;
    * from an expression resulting in a :class:`Matrix` or sparse matrix;
    * from a NumPy :class`ndarray`;
    * from a SciPy sparse matrix.

    You can also provide the following keyword parameters to the constructor:

    :param: *nnz*

       The number of nonzeros for which to allocate memory.

    :param: *shape*

       A tuple of two ints giving the shape of the matrix: (rows, cols).

    See also the constructor parameters inherited from :class:`Leaf`.

    Also inherits convenience functions for arithmetic; see
    :class:`MagicMethods`.

    """
    ndim = 2
    flushed = False
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY

    @classmethod
    def generate_fdm_laplace(cls, points_x, points_y, dtype=None, context=None):
        """Generates a sparse matrix obtained from a simple finite-difference
        discretization of the Laplace equation on the unit square
        (2-D).
        """
        A = cls(dtype=dtype, context=context)
        vcl_context = A.context.vcl_context
        generator = getattr(_v, "generate_fdm_laplace_" + vcl_statement_node_numeric_type_strings[A.statement_node_numeric_type])
        cpu_laplace = generator(points_x, points_y)
        A.cpu_leaf = cpu_laplace
        A.cpu_leaf.vcl_context = vcl_context
        return A

    @property
    def vcl_leaf_factory(self):
        """Derived classes should construct and return the C++ object
        representing the appropriate sparse matrix on the compute
        device by overriding this function.
        """
        raise NotImplementedError("This is only a base class!")

    def _init_leaf(self, args, kwargs):
        """Do general sparse-matrix-specific construction tasks, like setting
        up the shape, layout, dtype and host memory cache.
        """
        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
        else:
            shape = ()

        if shape and len(shape) != 2:
            raise TypeError("Any matrix can only be 2-dimensional!")

        if 'nnz' in kwargs.keys():
            nnz = kwargs['nnz']

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                raise TypeError("COL_MAJOR sparse layout not yet supported")
                self.layout = COL_MAJOR
            else:
                self.layout = ROW_MAJOR
        else:
            self.layout = ROW_MAJOR

        CONSTRUCT_FROM_DATA = False
        CONSTRUCT_FROM_SPMATRIX = False

        if len(args) == 0:
            # 0: empty -> empty
            if shape:
                def get_cpu_leaf(cpu_t):
                    return cpu_t(*shape)
            else:
                def get_cpu_leaf(cpu_t):
                    return cpu_t()

        elif len(args) == 1:
            if isinstance(args[0], tuple):
                # Then we construct from given data
                if nnz:
                    nnz = max(len(args[0][2])+1, nnz)
                else:
                    nnz = len(args[0][2])+1
                if shape:
                    shape = list(shape)
                    shape[0] = max(max(args[0][0])+1, shape[0])
                    shape[1] = max(max(args[0][1])+1, shape[1])
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(shape[0], shape[1], nnz)
                else:
                    # args[0] is (rows, cols, values), so ...
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(max(args[0][0])+1, max(args[0][1])+1,
                                     nnz)
                CONSTRUCT_FROM_DATA = True
                data = args[0]

            elif isinstance(args[0], Matrix):
                # 1: Matrix instance -> copy
                if not shape:
                    shape = args[0].shape
                elif shape != args[0].shape:
                    raise TypeError("Shapes not compatible")
                if self.dtype is None:
                    self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0].as_ndarray())

            elif isinstance(args[0], SparseMatrixBase):
                # 1: SparseMatrixBase instance -> copy
                if not shape:
                    shape = args[0].shape
                elif shape != args[0].shape:
                    raise TypeError("Shapes not compatible")
                if self.dtype is None:
                    self.dtype = args[0].dtype
                self.layout = args[0].layout
                def get_cpu_leaf(cpu_t):
                    return args[0].cpu_leaf

            elif isinstance(args[0], Node):
                # 1: Node instance -> get result and copy
                result = args[0].result
                if not shape:
                    shape = result.shape
                elif shape != result.shape:
                    raise TypeError("Shapes not compatible")
                if self.dtype is None:
                    self.dtype = result.dtype
                self.layout = result.layout
                if isinstance(result, SparseMatrixBase):
                    def get_cpu_leaf(cpu_t):
                        return result.cpu_leaf
                elif isinstance(result, Matrix):
                    def get_cpu_leaf(cpu_t):
                        return cpu_t(result.as_ndarray())
                else:
                    raise TypeError(
                        "Sparse matrix cannot be constructed thus")

            elif isinstance(args[0], ndarray):
                # 1: ndarray -> init and fill
                if not shape:
                    shape = args[0].shape
                elif shape != args[0].shape:
                    raise TypeError("Shapes not compatible")
                if self.dtype is None:
                    self.dtype = np_result_type(args[0])
                def get_cpu_leaf(cpu_t):
                    return cpu_t(args[0])

            elif isinstance(args[0], scipy_sparse_type):
                # 1: scipy sparse matrix -> init and fill
                spmatrix = args[0]
                if self.dtype is None:
                    self.dtype = np_result_type(spmatrix)
                if not shape:
                    shape = spmatrix.shape
                elif shape != spmatrix.shape:
                    raise TypeError("Given shapes not compatible")
                def get_cpu_leaf(cpu_t):
                    return cpu_t(shape[0], shape[1], spmatrix.nnz)
                CONSTRUCT_FROM_SPMATRIX = True

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
            self.dtype = self._context.default_dtype            
        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]

        try:
            self.cpu_leaf_type = getattr(
                _v,
                "cpu_sparse_matrix_" +
                vcl_statement_node_numeric_type_strings[
                    self.statement_node_numeric_type])
        except (KeyError, AttributeError):
            raise TypeError("dtype %s not supported" % self.statement_node_numeric_type)

        self.cpu_leaf = get_cpu_leaf(self.cpu_leaf_type)
        self.cpu_leaf.vcl_context = self._context.vcl_context

        if CONSTRUCT_FROM_DATA:
            def insert_entry(row, col, value):
                try:
                    self.cpu_leaf.insert_entry(row, col, value)
                except IndexError:
                    self.cpu_leaf.set_entry(row, col, value)
            x = list(map(insert_entry, data[0], data[1], data[2]))

        elif CONSTRUCT_FROM_SPMATRIX:
            def insert_entry(row, col):
                try:
                    self.cpu_leaf.insert_entry(int(row), int(col), spmatrix[row, col])
                except IndexError:
                    self.cpu_leaf.set_entry(int(row), int(col), spmatrix[row, col])
            x = list(map(insert_entry, spmatrix.nonzero()[0], spmatrix.nonzero()[1]))

        self.base = self

    @property
    def handle(self):
        """A tuple of :class:`backend.MemoryHandle` objects representing the
        storage of this object on the compute device.

        """
        if not self.flushed:
            self.flush()
        return self._handle

    @property
    def nonzeros(self):
        """A ``list`` of coordinates of the nonzero elements of the matrix.
        """
        return self.cpu_leaf.nonzeros

    @property
    def nnz(self):
        """The number of nonzero elements stored in the matrix, as an
        integer.
        """
        return self.cpu_leaf.nnz

    @property
    def size1(self):
        """The size of the first axis of the matrix.
        """
        return self.cpu_leaf.size1

    @property
    def size2(self):
        """The size of the second axis of the matrix.
        """
        return self.cpu_leaf.size2

    @property
    def size(self):
        """The flat size (area) of the matrix, as it would be in dense form.
        """
        return self.size1 * self.size2 # Flat size

    @property
    def shape(self):
        """The shape of the matrix as a 2-tuple, with one entry for each
        axis.
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
        """Returns the sparse matrix as a dense NumPy :class:`ndarray`.
        """
        return self.cpu_leaf.as_ndarray()

    def as_dense(self):
        """Returns the sparse matrix as a dense PyViennaCL :class:`Matrix`.
        """
        return Matrix(self, context = self._context)

    def as_lil_matrix(self):
        """Returns the sparse matrix as a SciPy :class:`lil_matrix`,
        if possible.
        """
        if not WITH_SCIPY:
            raise TypeError("SciPy not found, so this is unsupported")

        lil = spsparse.lil_matrix(self.shape, dtype=self.dtype)

        for nz in self.nonzeros:
            lil[nz[0], nz[1]] = self[nz[0], nz[1]]

        return lil

    @property
    def value(self):
        """Returns the sparse matrix as a SciPy :class:`lil_matrix`,
        if possible. If SciPy is not available, this resorts to constructing
        a dense NumPy ndarray.
        """
        if WITH_SCIPY:
            return self.as_lil_matrix()
        else:
            return self.as_ndarray()

    @property
    def vcl_leaf(self):
        """The underlying C++ ViennaCL object representing the matrix on the
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

    def insert(self, x, y, value):
        self.flushed = False
        if isinstance(value, ScalarBase):
            value = value.value
        self.cpu_leaf.insert_entry(x, y, np_result_type(self).type(value))

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

    def __repr__(self):
        out = "<pyviennacl.%s object of size (%d, %d) with %d nonzeros at 0x%x>" % (type(self).__name__, self.size1, self.size2, self.nnz, id(self))
        return out


class CompressedMatrix(SparseMatrixBase):
    """This class represents a sparse matrix on the ViennaCL compute
    device, in a compressed-row storage format.

    For information on construction, see the help for
    :class:`SparseMatrixBase`.
    """
    statement_node_subtype = _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_compressed_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle1),
                        backend.MemoryHandle(self._vcl_leaf.handle2))
        self.flushed = True


class CompressedCompressedMatrix(SparseMatrixBase):
    """A sparse square matrix in compressed sparse rows format optimized
    for the case that only a few rows carry nonzero entries.

    The difference from the 'standard' CSR format is that there is an
    additional array ``row_indices`` so that the i-th set of indices
    in the CSR layout refers to ``row_indices[i]``.
    [description adapted from the ViennaCL manual]

    For information on construction, see the help for
    :class:`SparseMatrixBase`.
    """
    #statement_node_subtype = _v.statement_node_subtype.COMPRESSED_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_compressed_compressed_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle1),
                        backend.MemoryHandle(self._vcl_leaf.handle2),
                        backend.MemoryHandle(self._vcl_leaf.handle3))
        self.flushed = True


class CoordinateMatrix(SparseMatrixBase):
    """This class represents a sparse matrix on the ViennaCL compute
    device, in a `coordinate` storage format: entries are stored as
    triplets ``(i, j, val)``, where ``i`` is the row index, ``j`` is
    the column index and ``val`` is the entry.

    For information on construction, see the help for
    :class:`SparseMatrixBase`.
    """
    statement_node_subtype = _v.statement_node_subtype.COORDINATE_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_coordinate_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle12),
                        backend.MemoryHandle(self._vcl_leaf.handle3))
        self.flushed = True


class ELLMatrix(SparseMatrixBase):
    """This class represents a sparse matrix on the ViennaCL compute
    device, in ELL storage format. In this format, the matrix is
    stored in a block of memory of size ``N`` by ``n_max``, where
    ``N`` is the number of rows of the matrix and ``n_max`` is the
    maximum number of nonzeros per row. Rows with fewer than ``n_max``
    entries are padded with zeros. In a second memory block, the
    respective column indices are stored.

    The ELL format is well suited for matrices where most rows have
    approximately the same number of nonzeros. This is often the case for
    matrices arising from the discretization of partial differential
    equations using e.g. the finite element method. On the other hand, the
    ELL format introduces substantial overhead if the number of nonzeros per
    row varies a lot.       [description adapted from the ViennaCL manual]

    For information on construction, see the help for
    :class:`SparseMatrixBase`.
    """
    statement_node_subtype = _v.statement_node_subtype.ELL_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_ell_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle2))
        self.flushed = True


class SlicedELLMatrix(SparseMatrixBase):
    """NB: This is an experimental format, with no support as of yet!

    TODO: construction with parameters C, sigma
    """
    #statement_node_subtype = _v.statement_node_subtype.SLICED_ELL_MATRIX_TYPE

    def flush(self):
        # TODO -- num_blocks!
        self._vcl_leaf = self.cpu_leaf.as_sliced_ell_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle1),
                        backend.MemoryHandle(self._vcl_leaf.handle2),
                        backend.MemoryHandle(self._vcl_leaf.handle3))
        self.flushed = True


class HybridMatrix(SparseMatrixBase):
    """This class represents a sparse matrix on the ViennaCL compute
    device, in a hybrid storage format, combining the higher
    performance of the ELL format for matrices with approximately the
    same number of entries per row with the higher flexibility of the
    compressed row format. The main part of the matrix is stored in
    ELL format and excess entries are stored in compressed row format.
    [description adapted from the ViennaCL manual]

    For information on construction, see the help for
    :class:`SparseMatrixBase`.
    """
    statement_node_subtype = _v.statement_node_subtype.HYB_MATRIX_TYPE

    def flush(self):
        self._vcl_leaf = self.cpu_leaf.as_hyb_matrix()
        self._handle = (backend.MemoryHandle(self._vcl_leaf.handle),
                        backend.MemoryHandle(self._vcl_leaf.handle2),
                        backend.MemoryHandle(self._vcl_leaf.handle3),
                        backend.MemoryHandle(self._vcl_leaf.handle4),
                        backend.MemoryHandle(self._vcl_leaf.handle5))
        self.flushed = True


# TODO: add ndarray flushing
class Matrix(Leaf):
    """This class represents a dense matrix object on the compute device, and 
    it can be constructed in a number of ways:

    * with no parameters, as an empty matrix;
    * from an integer tuple: produces an empty :class:`Matrix` of that shape;
    * from a tuple: first two values shape, third scalar value for each
      element;
    * from an ndarray of the correct dtype;
    * from a ViennaCL sparse matrix;
    * from a ViennaCL :class:`Matrix` instance (to make a copy);
    * from an expression resulting in a :class:`Matrix`;
    * from a PyOpenCL *Array*: producing a view onto the associated buffer;
    * from a host / OpenCL / CUDA buffer, wrapped in a
      :class:`backend.MemoryHandle` object.

    You can also provide the following keyword parameters to the constructor:

    :param: *layout* : either ROW_MAJOR ('C') or COL_MAJOR ('F')

       Layout of the matrix in memory. This corresponds to NumPy's *order*
       parameter. The default is ROW_MAJOR.

    :param: *shape* : 2-tuple of ints

       The shape of the matrix as a tuple of two ints: (rows, cols).

    And, if you are constructing fom a buffer via a
    :class:`backend.MemoryHandle` object:

    :param: *internal_shape* : 2-tuple of ints

       The shape of the matrix in memory, including any padding, as a
       tuple of two ints: (rows, cols). The default is the shape, or
       the next largest multiple of (128, 128), if the shape is not
       such a multiple.

    :param: *offset* : 2-tuple of ints

       The first int gives the starting index for the rows in the
       buffer of the target matrix, whilst the second int gives the
       starting index for the columns. Each should be given as a
       multiple of the size in bytes of each element of the matrix.
       The default is (0, 0).

    :param: *strides* : 2-tuple of ints

       The first int gives the number of bytes between each row, and the
       second gives the number between each column. Each should be given
       as multiple of the size in bytes of each element of the matrix. The
       default, if ROW_MAJOR, is ``(itemsize * internal_shape[0], itemsize)``;
       if COL_MAJOR, ``(itemsize, itemsize * internal_shape[1])``.

    See also the constructor parameters inherited from :class:`Leaf`.

    Also inherits convenience functions for arithmetic; see
    :class:`MagicMethods`.

    Thus, to construct a 5-by-5 column-major Matrix instance with a numeric 
    data type of ``float32`` (C++ ``float``) and each element being equal to
    ``3.141``, type:

      >>> import pyviennacl as p
      >>> mat = p.Matrix(5, 5, 3.141, dtype=p.float32, layout=p.COL_MAJOR)
      >>> print(mat)
      [[ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]
       [ 3.14100003  3.14100003  3.14100003  3.14100003  3.14100003]]

    """
    ndim = 2
    statement_node_type_family = _v.statement_node_type_family.MATRIX_TYPE_FAMILY
    statement_node_subtype = _v.statement_node_subtype.DENSE_MATRIX_TYPE

    def _init_leaf(self, args, kwargs):
        """Construct the underlying ViennaCL vector object according to the
        given arguments and types.
        """
        if 'shape' in kwargs.keys():
            if len(kwargs['shape']) != 2:
                raise TypeError("Matrix can only have a 2-d shape")
            shape = kwargs['shape']
        else:
            shape = ()

        if 'layout' in kwargs.keys():
            if kwargs['layout'] == COL_MAJOR:
                self.layout = COL_MAJOR
            else:
                self.layout = ROW_MAJOR
        else:
            self.layout = ROW_MAJOR

        if len(args) == 0:
            if shape:
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], shape[1], self._context.vcl_context)
            else:
                def get_leaf(vcl_t):
                    return vcl_t(self._context.vcl_context)

        elif len(args) == 1:
            if isinstance(args[0], MagicMethods):
                if issubclass(args[0].result_container_type,
                              SparseMatrixBase):
                    if not shape:
                        shape = args[0].shape
                    elif shape != args[0].shape:
                        raise TypeError("Shapes not compatible!")
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    self.layout = args[0].result.layout
                    def get_leaf(vcl_t):
                        return vcl_t(args[0].result.as_ndarray(),
                                     self._context.vcl_context)

                elif issubclass(args[0].result_container_type, Matrix):
                    if args[0].context.domain is not self._context.domain:
                        raise TypeError("TODO Can only construct from objects with same memory domain")
                    if not shape:
                        shape = args[0].shape
                    elif shape != args[0].shape:
                        raise TypeError("Shapes not compatible!")
                    if self.dtype is None:
                        self.dtype = args[0].result.dtype
                    self.layout = args[0].result.layout
                    def get_leaf(vcl_t):
                        return vcl_t(args[0].result.vcl_leaf)

                else:
                    raise TypeError(
                        "Matrix cannot be constructed in this way")

            elif isinstance(args[0], tuple) or isinstance(args[0], list):
                if not shape:
                    shape = args[0]
                elif list(shape) != list(args[0]):
                    raise TypeError("Shapes not compatible!")
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], shape[1], self._context.vcl_context)

            elif isinstance(args[0], tuple(vcl_matrix_base_types.values())):
                # Crudely find the dtype of the argument
                test_dtype = None
                for key in vcl_matrix_base_types.keys():
                    if isinstance(args[0], vcl_matrix_base_types[key]):
                        test_dtype = dtype(getattr(numpy, HostScalarTypes_inverse[vcl_statement_node_numeric_type_strings_inverse[key]]))
                        break

                if test_dtype is None:
                    raise TypeError("Could not deduce dtype of argument")
                if self.dtype is None:
                    self.dtype = test_dtype
                elif test_dtype != self.dtype:
                    raise TypeError("Cannot convert dtypes")

                if args[0].row_major and self.layout != ROW_MAJOR:
                    raise TypeError("Cannot convert matrix layouts")
                if not args[0].row_major and self.layout != COL_MAJOR:
                    raise TypeError("Cannot convert matrix layouts")

                if backend.vcl_memory_types[args[0].memory_domain] is not self._context.domain:
                    raise TypeError("TODO Can only construct from objects with same memory domain")

                def get_leaf(vcl_t):
                    return args[0]

            elif isinstance(args[0], ndarray):
                if not shape:
                    shape = args[0].shape
                elif shape != args[0].shape:
                    raise TypeError("Shapes not compatible!")
                if self.dtype is None:
                    self.dtype = args[0].dtype
                def get_leaf(vcl_t):
                    return vcl_t(args[0], self._context.vcl_context)

            elif isinstance(args[0], tuple(mem_handle_types)):
                mem_handle = args[0]
                WITH_CL_ARRAY = False
                if WITH_OPENCL:
                    if isinstance(mem_handle, cl.MemoryObject):
                        mem_handle = backend.MemoryHandle(mem_handle)
                    elif isinstance(mem_handle, cl.array.Array):
                        cl_array = mem_handle
                        if len(cl_array.shape) > 2:
                            raise TypeError("Can only construct a Matrix from a 2-D array!")
                        mem_handle = backend.MemoryHandle(cl_array.base_data)
                        WITH_CL_ARRAY = True

                if self.dtype is None:
                    if WITH_CL_ARRAY:
                        self.dtype = np_result_type(cl_array)
                    else:
                        raise TypeError("You must set the dtype if constructing from a MemoryHandle")

                if not shape:
                    if WITH_CL_ARRAY:
                        shape = cl_array.shape
                    else:
                        raise TypeError("You must provide the shape of the matrix")

                if mem_handle.context is None:
                    mem_handle.context = self._context

                if self.layout == ROW_MAJOR:
                    is_row_major = True
                else:
                    is_row_major = False

                if 'internal_shape' in kwargs.keys():
                    internal_shape = kwargs['internal_shape']
                elif WITH_CL_ARRAY:
                    internal_shape = cl_array.shape
                else:
                    internal_shape = (int(mem_handle.raw_size / (shape[1] * self.itemsize)),
                                      int(mem_handle.raw_size / (shape[0] * self.itemsize)))

                if 'strides' in kwargs.keys():
                    if is_row_major:
                        vcl_strides = (int(kwargs['strides'][0] / (internal_shape[0] * self.itemsize)),
                                       int(kwargs['strides'][1] / self.itemsize))
                    else:
                        vcl_strides = (int(kwargs['strides'][0] / self.itemsize),
                                       int(kwargs['strides'][1] / (internal_shape[1] * self.itemsize)))
                elif WITH_CL_ARRAY:
                    if is_row_major:
                        vcl_strides = (int(cl_array.strides[0] / (internal_shape[0] * self.itemsize)),
                                       int(cl_array.strides[1] / self.itemsize))
                    else:
                        vcl_strides = (int(cl_array.strides[0] / self.itemsize),
                                       int(cl_array.strides[1] / (internal_shape[1] * self.itemsize)))
                else:
                    vcl_strides = (1, 1)

                if 'offset' in kwargs.keys():
                    offset = (int(kwargs['offset'][0] / self.itemsize),
                              int(kwargs['offset'][1] / self.itemsize))
                elif WITH_CL_ARRAY:
                    if is_row_major:
                        offset = (int(cl_array.offset / self.itemsize), 0)
                    else:
                        offset = (0, int(cl_array.offset / self.itemsize))
                else:
                    offset = (0,0)

                def get_leaf(vcl_t):
                    base_t = vcl_t.__bases__[0]
                    return base_t(mem_handle.vcl_handle, shape[0], offset[0], vcl_strides[0], internal_shape[0], shape[1], offset[1], vcl_strides[1], internal_shape[1], is_row_major)

            else:
                # This doesn't do any dtype checking, so beware...
                def get_leaf(vcl_t):
                    return args[0]

        elif len(args) == 2:
            if isinstance(args[0], tuple) or isinstance(args[0], list):
                if not shape:
                    shape = args[0]
                elif list(shape) != list(args[0]):
                    raise TypeError("Shapes not compatible!")
                if self.dtype is None:
                    self.dtype = np_result_type(args[1])
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], shape[1], args[1],
                                 self._context.vcl_context)

            else:
                if not shape:
                    shape = args
                elif list(shape) != list(args):
                    raise TypeError("Shapes not compatible!")
                def get_leaf(vcl_t):
                    return vcl_t(shape[0], shape[1], self._context.vcl_context)

        elif len(args) == 3:
            if not shape:
                shape = args[:2]
            elif list(shape) != list(args[:2]):
                raise TypeError("Shapes not compatible!")
            if self.dtype is None:
                self.dtype = np_result_type(args[2])
            def get_leaf(vcl_t):
                return vcl_t(shape[0], shape[1], args[2],
                             self._context.vcl_context)

        else:
            raise TypeError("Matrix cannot be constructed in this way")

        if self.dtype is None: # ie, still None, even after checks -- so guess
            self.dtype = self._context.default_dtype

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
        self._handle = (backend.MemoryHandle(self.vcl_leaf.handle),)

    @property
    def size1(self):
        return self.vcl_leaf.size1

    @property
    def size2(self):
        return self.vcl_leaf.size2

    @property
    def size(self):
        return self.size1 * self.size2

    @property
    def shape(self):
        return (self.size1, self.size2)

    @property
    def internal_size1(self):
        return self.vcl_leaf.internal_size1

    @property
    def internal_size2(self):
        return self.vcl_leaf.internal_size2

    @property
    def internal_size(self):
        return self.internal_size1 * self.internal_size2

    @property
    def internal_shape(self):
        return (self.internal_size1, self.internal_size2)

    @property
    def strides(self):
        if self.layout == ROW_MAJOR:
            return (self.vcl_leaf.stride1 * self.internal_size1 * self.itemsize, self.vcl_leaf.stride2 * self.itemsize)
        else:
            return (self.vcl_leaf.stride1 * self.itemsize, self.vcl_leaf.stride2 * self.itemsize * self.internal_size2)

    @property
    def offset(self):
        return (self.vcl_leaf.start1 * self.strides[0], self.vcl_leaf.start2 * self.strides[1])

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
                    if abs(key[0]) >= self.shape[0]:
                        raise IndexError("Index larger than first axis")
                    if key[0] < 0:
                        key[0] += self.shape[0]
                    # Choose from row
                    if isinstance(key[1], int):
                        #  (int, int) -> scalar
                        if abs(key[1]) >= self.shape[1]:
                            raise IndexError("Index larger than second axis")
                        if key[1] < 0:
                            key[1] += self.shape[1]
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
                        if abs(key[1]) >= self.shape[1]:
                            raise IndexError("Index larger than second axis")
                        if key[1] < 0:
                            key[1] += self.shape[1]
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
            return self[key, :]
        else:
            raise IndexError("Did not understand key")

    def as_opencl_kernel_operands(self):
        """Returns a representation of the current object sufficient for
        passing to PyOpenCL for executing a custom kernel.

        In this case, this means a list consisting of the buffer for this
        object and its sizes.
        """
        return [self.handle[0].buffer,
                uint32(self.internal_size1), uint32(self.internal_size2)]

    def as_vector(self, copy=False):
        """TODO docstring

        NB: param copy is meaningless if self.size != self.internal_size!
        NB: copies rows if ROW_MAJOR, cols if COL_MAJOR
        """
        if self.offset[1] != 0:
            raise ValueError("Offset in the second index must be 0")
        if self.size == self.internal_size:
            if copy:
                tmp = self.copy()
            else:
                tmp = self
            if self.layout == ROW_MAJOR:
                new_vector = Vector(tmp.handle[0].buffer, size=tmp.size, offset=tmp.offset[0], stride=tmp.strides[1], dtype=tmp.dtype, context=tmp.context)
            else:
                new_vector = Vector(tmp.handle[0].buffer, size=tmp.size, offset=tmp.offset[0], stride=tmp.strides[0], dtype=tmp.dtype, context=tmp.context)
        else:
            new_cpu_vector = zeros(self.size, dtype=self.dtype)
            if self.layout == ROW_MAJOR:
                for index in range(self.shape[0]):
                    new_cpu_vector[index*self.shape[1]:(index+1)*self.shape[1]] = self[index, :].as_ndarray()
            else:
                for index in range(self.shape[1]):
                    new_cpu_vector[index*self.shape[0]:(index+1)*self.shape[0]] = self[:, index].as_ndarray().flatten()
            new_vector = Vector(new_cpu_vector, context=self.context)
        return new_vector

    #def clear(self):
    #    """
    #    Set every element of the matrix to 0.
    #    """
    #    return self.vcl_leaf.clear()

    @property
    def T(self):
        """Return the matrix transpose.
        """
        return Trans(self)
    trans = T


class Node(MagicMethods):
    """This is the base class for all nodes in the ViennaCL expression
    tree. A Node is any n-ary operation, such as addition. This class
    provides logic for expression tree construction and result type
    deduction, in order that expression statements can be executed
    correctly.

    If you're extending the ViennaCL core by adding an operation and
    want support for it in Python, then you should derive from this
    class.
    """

    statement_node_type_family = _v.statement_node_type_family.INVALID_TYPE_FAMILY
    statement_node_subtype = _v.statement_node_subtype.INVALID_SUBTYPE
    statement_node_numeric_type = _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE
    operation_node_type_family = _v.operation_node_type_family.OPERATION_INVALID_TYPE_FAMILY
    operation_node_type = _v.operation_node_type.OPERATION_INVALID_TYPE
    operands = []
    flushed = False

    def __init__(self, *args):
        """Take the given operand(s) to an appropriate representation for
        this operation, and deduce the result_type. Construct a
        ViennaCL statement_node object representing this information,
        ready to be inserted into an expression statement.
        """
        if len(args) == 1:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY
        elif len(args) == 2:
            self.operation_node_type_family = _v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY
        else:
            raise TypeError("Only unary or binary nodes supported currently")

        def fix_operand(arg):
            return util.fix_operand(arg, self)
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

        self.statement_node_type_family = _v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY # Finish initialising

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._delete_operands()

    def _node_init(self):
        pass

    def _vcl_node_init(self):
        # At the moment, ViennaCL does not do dtype promotion, so check that
        # the operands all have the same dtype.
        if len(self.operands) > 1:
            if dtype(self.operands[0]) != dtype(self.operands[1]):
                if issubclass(self.operands[1].result_container_type, ScalarBase):
                    fix = self.operands[1].result_container_type(np_result_type(self.operands[0]).type(self.operands[1].value))
                    self.operands[1] = fix
                elif issubclass(self.operands[0].result_container_type, ScalarBase):
                    fix = self.operands[0].result_container_type(np_result_type(self.operands[1]).type(self.operands[0].value))
                    self.operands[1] = fix
                else:
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
            # Set up the ViennaCL statement_node with one operand,
            # and set rhs to INVALID
            self.vcl_node = _v.statement_node(
                self.operands[0].statement_node_type_family,   # lhs
                self.operands[0].statement_node_subtype,       # lhs
                self.operands[0].statement_node_numeric_type,  # lhs
                self.operation_node_type_family,               # op
                self.operation_node_type,                      # op
                _v.statement_node_type_family.INVALID_TYPE_FAMILY,    # rhs
                _v.statement_node_subtype.INVALID_SUBTYPE,            # rhs
                _v.statement_node_numeric_type.INVALID_NUMERIC_TYPE)  # rhs

    def _delete_operands(self):
        del self.operands
        self.operands = []

    def _test_init(self):
        layout_test = self.layout # NB QUIRK

    def __getitem__(self, key):
        return self.result[key]

    def __setitem__(self, key, value):
        self.result[key] = value

    def get_vcl_operand_setter(self, operand):
        """This function returns the correct function for setting the
        underlying ViennaCL statement_node object's operand(s) in the
        correct way: each different container type and dtype are
        mapped onto a different set_operand_to function in the
        underlying object, in order to avoid type ambiguity at the
        Python/C++ interface.
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
        """The complexity of the ViennaCL expression, given as the number of
        :class:`Node` instances in the expression tree.
        """
        complexity = 1
        for op in self.operands:
            complexity += op.complexity
        return complexity

    @property
    def operand_types_string(self):
        """A tuple of strings representing the names of the types of the
        operands.

        """
        types = []
        for o in self.operands:
            try:
                t = o.result_container_type.__name__
            except AttributeError:
                t = 'HostScalar'
            types.append(t)
        return tuple(types)

    @property
    def result_container_type(self):
        """Determine the container type (ie, :class:`Scalar`, :class:`Vector`,
        etc) needed to store the result of the operation encoded by
        this :class:`Node`. If the operation has some effect (eg,
        in-place), but does not produce a distinct result, then return
        :class:`NoResult`. If the operation is not supported for the
        given operand types, then return ``None``.
        """
        if self.flushed:
            return type(self._result)
        if len(self.result_types) < 1:
            return NoResult
        try: return self.result_types[self.operand_types_string]
        except KeyError: return None

    @property
    def dtype(self):
        """Determine the dtype of the scalar element(s) of the result of the
        operation encoded by this :class:`Node`, according to the
        NumPy type promotion rules.
        """
        if self.flushed:
            return np_result_type(self._result)

        dtypes = tuple(map(lambda x: x.dtype, self.operands))
        if len(dtypes) == 1:
            return np_result_type(dtypes[0])
        if len(dtypes) == 2:
            return np_result_type(dtypes[0], dtypes[1])

    @property
    def layout(self):
        """Recursively determine the storage layout for the result type, if
        the result is a :class:`Matrix`.

        Notably, this ensures that any :class:`Matrix` operands have
        the same layout, since this is a condition of all ViennaCL
        operations, except for the matrix-matrix product.
        """
        if self.flushed:
            return self._result.layout

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
        """Determine the maximum number of dimensions required to store the
        result of any operation on the given operands.

        This can be overridden by the particular :class:`Node`
        subclass, in order to compute the correct size for the result
        container.
        """
        if self.flushed:
            return self._result.ndim

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
        """Determine the maximum size of any axis required to store the
        result of any operation on the given operands.

        This can be overridden by the particular :class:`Node`
        subclass, in order to compute the correct size for the result
        container.
        """
        if self.flushed:
            return max(self._result.shape)

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
        """Determine the upper-bound shape of the object needed to store the
        result of any operation on the given operands. The ``len`` of
        this tuple is the number of dimensions, with each element of
        the tuple being the upper-bound size of the corresponding
        dimension.

        If the shape is set manually, then this routine is overridden, and
        the manually set value is returned.

        It is advisable to override this; you will get a warning otherwise!
        """
        if self.flushed:
            return self._result.shape

        try:
            if isinstance(self._shape, tuple):
                return self._shape
        except: pass

        log.warning("Node class %s does not define its own shape; guessing.." % type(self).__name__)

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

    @property
    def internal_shape(self):
        return self.result.internal_shape

    @property
    def offset(self):
        return self.result.offset

    @property
    def strides(self):
        return self.result.strides

    def express(self, statement=""):
        """Produce a human-readable representation of the expression graph
        including all nodes and leaves connected to this one, which
        constitutes the root node.
        """
        if self.result_container_type is None:
            result_expression = "None"
        else:
            result_expression = self.result_container_type.__name__
        result_expression += ":" + str(dtype(self))

        statement += type(self).__name__ + "("
        if self.flushed:
            statement += "[flushed])=>" + result_expression
        else:
            for op in self.operands:
                statement = op.express(statement) + ", "
            statement = statement[:-2] + ")=>" + result_expression
        return statement

    @property
    def result(self):
        """The result of computing the operation represented by this
        :class:`Node` instance. Returns the cached result if there is
        one, otherwise executes the corresponding expression, caches
        the result, and returns that.
        """
        if not self.flushed:
            self.execute()
        return self._result

    @property
    def vcl_leaf(self):
        return self.result.vcl_leaf

    def execute(self):
        """Execute the expression tree taking this instance as the root, and
        then cache and return the result.
        """
        if self.flushed:
            log.warning("Node already flushed, so returning cached result")
            return self._result

        s = Statement(self)
        self._result = s.execute()
        self.flushed = True
        self._delete_operands()
        return self._result

    @property
    def value(self):
        """The value of the result of computing the operation represented by
        this :class:`Node`; if the result is a :class:`Vector` or
        :class:`Matrix`, then the type is a NumPy :class:`ndarray`;
        if the result is a sparse matrix, then the type is a SciPy
        :class:`lil_matrix`; otherwise, a scalar is returned.

        """
        return self.result.value

    @property
    def handle(self):
        """Returns the memory handle of the result of the operation.
        
        Be aware that this incurs the execution of the statement, if this
        has not yet occurred.
        """
        return self.result.handle

    @property
    def context(self):
        if self.flushed:
            return self._result.context

        # NB: This assumes all operands have the same context (TODO: fix?)
        return self.operands[0].context

    def as_ndarray(self):
        """Return the value of computing the operation represented by this
        Node as a NumPy :class:`ndarray`.
        """
        return array(self.value, dtype=self.dtype)

    def as_opencl_kernel_operands(self):
        """Returns a representation of the current object sufficient for
        passing to PyOpenCL for executing a custom kernel.
        """
        return self.result.as_opencl_kernel_operands()


class CustomNode(Node):
    """TODO docstring
    """
    result_types = {}
    kernels = {}

    def __init__(self, *args):
        """TODO docstring
        """
        def fix_operand(args):
            return util.fix_operand(args, self)
        self.operands = list(map(fix_operand, args))
        self.statement_node_type_family = self.result_container_type.statement_node_type_family
        self.statement_node_subtype = self.result_container_type.statement_node_subtype
        self.statement_node_numeric_type = HostScalarTypes[self.dtype.name]
        for op in self.operands:
            if op.context != self.context and not isinstance(op, HostScalar):
                raise TypeError("Operands must all have the same context")
        self._compile_kernels()

    def _compile_kernels(self):
        # TODO: global cache, so every instantiation doesn't necessitate a
        #       full compile
        self.compiled_kernels = {}
        for domain in self.kernels.keys():
            self.compiled_kernels[domain] = {}
            for op_type in self.kernels[domain]:
                kernel = self.kernels[domain][op_type]
                if domain is backend.OpenCLMemory and isinstance(kernel, str):
                    prg = cl.Program(self.context.sub_context, kernel)
                    prg = prg.build()
                    try:
                        built_kernel = prg.all_kernels()[0]
                    except IndexError:
                        log.warning("Failed to build kernel for domain %s and operand types %s" % (domain, op_type))
                        built_kernel = noop
                    self.compiled_kernels[domain][op_type] = built_kernel
                else:
                    self.compiled_kernels[domain][op_type] = kernel

    def execute(self):
        """TODO docstring
        """
        try:
            self._kernel = self.compiled_kernels[self.context.domain][self.operand_types_string]
        except AttributeError:
            log.error("No kernel for memory domain %s and operand types %s"
                    % (self.context.domain, self.operand_types_string))
        result = self.result_container_type(
            shape = self.shape,
            dtype = self.dtype,
            layout = self.layout,
            context = self.context)
        operands = self.operands + [result]
        if self.context.domain is backend.OpenCLMemory:
            if isinstance(self._kernel, cl.Kernel):
                self._opencl_execute_kernel(*operands)
            else:
                self._kernel(*operands)
        else:
            self._kernel(*operands)
        self._result = result
        self.flushed = True

    def _opencl_execute_kernel(self, *operands):
        """TODO docstring
        """
        args = [x.as_opencl_kernel_operands() for x in operands]
        args = itertools.chain(*args)
        self._kernel(self.context.current_queue,
                     self.opencl_global_size, self.opencl_local_size, *args)

    @property
    def opencl_global_size(self):
        """TODO docstring -- override
        """
        log.warning("No global work size specified; using shape")
        return self.shape

    @property
    def opencl_local_size(self):
        """TODO docstring -- override
        """
        log.warning("No local work size specific; using None")
        return None


class Norm_1(Node):
    """Represent the computation of the L^1-norm of a :class:`Vector`.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_1_TYPE
    shape = ()


class Norm_2(Node):
    """Represent the computation of the L^2-norm of a :class:`Vector`.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_2_TYPE
    shape = ()


class Norm_Inf(Node):
    """Represent the computation of the L^inf-norm of a :class:`Vector`.
    """
    result_types = {
        ('Vector',): HostScalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_NORM_INF_TYPE
    shape = ()


class Neg(Node):
    """Represent ``-x``; the negation of the operand.
    """
    result_types = {
        ('Vector',): Vector,
        ('Matrix',): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_MINUS_TYPE

    def _node_init(self):
        self.shape = self.operands[0].shape


class ElementAbs(Node):
    """Represent the elementwise computation of ``abs`` on an object.
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
    """Represent the elementwise computation of ``acos`` on an object.
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
    """Represent the elementwise computation of ``asin`` on an object.
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
    """Represent the elementwise computation of ``atan`` on an object.
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
    """Represent the elementwise computation of ``ceil`` on an object.
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
    """Represent the elementwise computation of ``cos`` on an object.
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
    """Represent the elementwise computation of ``cosh`` on an object.
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
    """Represent the elementwise computation of ``exp`` on an object.
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
    """Represent the elementwise computation of ``fabs`` on an object.
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
    """Represent the elementwise computation of ``floor`` on an object.
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
    """Represent the elementwise computation of ``log`` (base e) on an
    object.
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
    """Represent the elementwise computation of ``log`` (base 10) on an
    object.
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
    """Represent the elementwise computation of ``sin`` on an object.
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
    """Represent the elementwise computation of ``sinh`` on an object.
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
    """Represent the elementwise computation of ``sqrt`` on an object.
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
    """Represent the elementwise computation of ``tan`` on an object.
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
    """Represent the elementwise computation of ``tanh`` on an object.
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
    """Represent the computation of the matrix transpose.
    """
    result_types = {
        ('Matrix',): Matrix,
    }
    operation_node_type = _v.operation_node_type.OPERATION_UNARY_TRANS_TYPE

    def _node_init(self):
        self.shape = (self.operands[0].shape[1],
                      self.operands[0].shape[0])


class Assign(Node):
    """Represent the assignment (copy) of one object's content to
    another.
    
    For example: ``x = y`` is represented by ``Assign(x, y)``.
    """
    result_types = {}
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot assign two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class InplaceAdd(Assign):
    """Represent the computation of the in-place addition of one object
    to another.
    
    Derives from :class:`Assign` rather than directly from
    :class:`Node` because in-place operations are mathematically
    similar to assignation.
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
    """Represent the computation of the in-place subtraction of one
    object to another.
    
    Derives from :class:`Assign` rather than directly from
    :class:`Node` because in-place operations are mathematically
    similar to assignation.
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
    """Represent the addition of one object to another.
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
    """Represent the subtraction of one object from another.
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
    """Represents the multiplication of one object by another.

    The semantics are as follows:

    * Scalar by scalar -> scalar;
    * scalar by vector -> scaled vector;
    * scalar by matrix -> scaled matrix;
    * vector by vector -> undefined;
    * vector by matrix -> undefined;
    * matrix by vector -> matrix-vector product;
    * matrix by matrix -> matrix-matrix product.
    
    The concern in defining these semantics has been to preserve the
    dimensionality of the operands in the result. The :class:`Mul`
    class does not map directly onto the ``*`` operator for every class.
    """
    result_types = {
        # OPERATION_BINARY_MAT_MAT_PROD_TYPE
        ('Matrix', 'Matrix'): Matrix,
        ('CompressedMatrix', 'CompressedMatrix'): Matrix,
        ('CoordinateMatrix', 'CoordinateMatrix'): Matrix,
        ('ELLMatrix', 'ELLMatrix'): Matrix,
        ('HybridMatrix', 'HybridMatrix'): Matrix,

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
        elif self.operands[0].result_container_type == Vector: # Vector * ...
            if self.operands[1].result_container_type == Scalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape
            elif self.operands[1].result_container_type == HostScalar:
                self.operation_node_type = _v.operation_node_type.OPERATION_BINARY_MULT_TYPE
                self.shape = self.operands[0].shape


class Div(Node):
    """Represents the division of a :class:`Matrix` or :class:`Vector` by
    a scalar.
    """
    result_types = {
        ('Vector', 'Scalar'): Vector,
        ('Vector', 'HostScalar'): Vector,
        ('Matrix', 'Scalar'): Matrix,
        ('Matrix', 'HostScalar'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_DIV_TYPE

    def _node_init(self):
        for x in self.operands:
            if x.result_container_type != ScalarBase:
                self.shape = x.shape
                break


class ElementPow(Node):
    """Represents the elementwise exponentiation of one object by another
    of the same type.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_POW_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot ElementProd two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class ElementProd(Node):
    """Represents the elementwise multiplication of one object by another
    of the same type.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_PROD_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot ElementProd two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class ElementDiv(Node):
    """Represents the elementwise multiplication of one object by another
    of the same type.
    """
    result_types = {
        ('Vector', 'Vector'): Vector,
        ('Matrix', 'Matrix'): Matrix
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_ELEMENT_DIV_TYPE

    def _node_init(self):
        if self.operands[0].shape != self.operands[1].shape:
            raise TypeError("Cannot ElementDiv two differently shaped objects! %s" % self.express())
        self.shape = self.operands[0].shape


class Dot(Node):
    """Represents the computation of the inner (dot) product of two
    vectors.
    """
    result_types = {
        ('Vector', 'Vector'): Scalar
    }
    operation_node_type = _v.operation_node_type.OPERATION_BINARY_INNER_PROD_TYPE
    shape = ()


class OrderType(object):
    def __init__(*args):
        raise TypeError("This class is not supposed to be instantiated")

class SequentialOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.SEQUENTIAL

class IndependentOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.INDEPENDENT


class StatementsTuple(object):
    vcl_statements_tuple = None

    def __init__(self, statements, order = SequentialOrder):
        if not isinstance(statements, list):
            statements = [statements]
        def to_vcl_statement(s):
            if isinstance(s, Node):
                return Statement(s).vcl_statement
            else:
                return s.vcl_statement
        vcl_statements = list(map(to_vcl_statement, statements))
        self.order = order
        self.vcl_tuple = _v.statements_tuple(vcl_statements, order.vcl_order)

class Statement(object):
    """This class represents the ViennaCL `statement` corresponding to an
    expression graph. It employs type deduction information to
    calculate the resultant types, and generates the appropriate
    underlying ViennaCL C++ object.
    """
    statement = []  # A list to hold the flattened expression tree
    vcl_statement = None # Will reference the ViennaCL statement object

    def __init__(self, root):
        """Given a :class:`Node` instance, return an object representing the
        ViennaCL statement of the corresponding expression graph, as
        connected to the given root node.

        If the given root node is not an instance of :class:`Assign`
        type, then a temporary object is constructed to store the
        result of executing the expression, and then a new ``Assign``
        instance is created, representing the assignation of the
        result of the expression to the new temporary.  The new
        ``Assign`` node is then taken to be the root node of the
        graph, having transposed the rest.
        """
        if not isinstance(root, Node):
            raise RuntimeError("Statement must be initialised on a Node")

        next_node = [] # Holds nodes as we travel down the tree

        # Test to see that we can actually do the operation
        if not root.result_container_type:
            raise TypeError("Unsupported expression: %s" %(root.express()))

        # If the root node is not an Assign instance, then construct a
        # temporary to hold the result.
        if isinstance(root, Assign):
            self.result = root.operands[0]
        else:
            self.result = root.new_instance()
            top = Assign(self.result, root)
            next_node.append(top)

        next_node.append(root)
        self.statement = []
        # Flatten the tree
        for n in next_node:
            op_num = 0
            for operand in n.operands:
                # If operand is a CustomNode, then we treat it as a Leaf here,
                # since CustomNode is not mapped to a ViennaCL statement node
                if isinstance(operand, Node) and not isinstance(operand, CustomNode) and not operand.flushed:
                    #if op_num == 0 and len(n.operands) > 1:
                    #    # ViennaCL cannot cope with complex LHS
                    #    operand = operand.result
                    #    n.operands[0] = operand
                    #    n._vcl_node_init()
                    #else:
                    next_node.append(operand)
                if isinstance(operand, Leaf) or isinstance(operand, CustomNode):
                    if (operand.context != self.result.context
                        and not isinstance(operand, HostScalar)
                        and not isinstance(self.result, HostScalar)):
                        raise TypeError(
                            "All objects in statement must have same context: %s"
                            % (operand.express()))
                op_num += 1
            append_node = not isinstance(n, CustomNode)
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
                    if operand.flushed or isinstance(operand, CustomNode):
                        # NB: CustomNode instances are dispatched here,
                        #     or the cached result used
                        n.get_vcl_operand_setter(operand.result)(
                            op_num, operand.result.vcl_leaf)
                    else:
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

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for n in self.statement:
            if n.flushed: n._delete_operands()
        del self.statement
        del self.vcl_statement
        del self.result

    def execute(self):
        """Execute the statement -- don't do anything else -- then return the
        result (if any).
        """
        if self.result.context.domain is backend.OpenCLMemory:
            vcl.set_active_context(self.result.context)
        try:
            self.vcl_statement.execute()
        except RuntimeError:
            log.error("EXCEPTION EXECUTING: %s" %(self.statement[0].express()))
            raise
        return self.result
        



__all__ = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
           'uint64', 'float16', 'float32', 'float64',
           'WITH_SCIPY', 'WITH_OPENCL',
           'ROW_MAJOR', 'COL_MAJOR', 'HostScalarTypes',
           'NoResult', 'MagicMethods', 'View', 'Leaf', 'ScalarBase',
           'HostScalar', 'Scalar', 'Vector', 'Matrix', 'SparseMatrixBase',
           'CompressedMatrix', 'CoordinateMatrix', 'ELLMatrix',
           'HybridMatrix', #'CompressedCompressedMatrix', 'SlicedELLMatrix',
           'Node', 'CustomNode', 'Norm_1', 'Norm_2', 'Norm_Inf', 'Neg',
           'ElementAbs', 'ElementAcos', 'ElementAsin', 'ElementAtan',
           'ElementCeil', 'ElementCos', 'ElementCosh', 'ElementExp',
           'ElementFabs', 'ElementFloor', 'ElementLog', 'ElementLog10',
           'ElementSin', 'ElementSinh', 'ElementSqrt', 'ElementTan',
           'ElementTanh', 'Trans', 'Assign', 'InplaceAdd', 'InplaceSub',
           'Add', 'Sub', 'Mul', 'Div',
           'ElementPow', 'ElementProd', 'ElementDiv', 'Dot',
           'Statement']

