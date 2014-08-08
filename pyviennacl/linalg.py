from pyviennacl import backend, tags, _viennacl as _v
from pyviennacl.pycore import (Matrix, SparseMatrixBase, ScalarBase, Vector,
                               Node, MagicMethods, Mul,
                               vcl_statement_node_numeric_type_strings)
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)
import logging

log = logging.getLogger(__name__)

def plane_rotation(vec1, vec2, alpha, beta):
    """
    Computes (vec1, vec2) <- (alpha*vec1+beta*vec2, -beta*vec1+alpha*vec2)
    
    Parameters
    ----------
    vec1 : Vector
    vec2 : Vector
    alpha : any Python, NumPy or PyViennaCL scalar (real or integer)
    beta : any Python, NumPy or PyViennaCL scalar (real or integer)

    Returns
    -------
    None

    Notes
    -----
    The dtypes of the parameters must match.

    Operates in-place on vec1 and vec2.
    """
    # Do an assortment of type and dtype checks...
    if isinstance(vec1, Node):
        vec1 = vec1.result
    if isinstance(vec2, Node):
        vec2 = vec2.result
    if isinstance(alpha, Node):
        alpha = alpha.result
    if isinstance(beta, Node):
        beta = beta.result
    if isinstance(vec1, Vector):
        x = vec1.vcl_leaf
        if isinstance(vec2, Vector):
            if vec1.dtype != vec2.dtype:
                raise TypeError("Vector dtypes must be the same")
            y = vec2.vcl_leaf
        else:
            y = vec2
    else:
        x = vec1
        if isinstance(vec2, Vector):
            y = vec2.vcl_leaf
        else:
            y = vec2

    if isinstance(alpha, ScalarBase):
        if isinstance(vec1, Vector):
            if alpha.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        a = alpha.value
    else:
        a = alpha

    if isinstance(beta, ScalarBase):
        if isinstance(vec1, Vector):
            if beta.dtype != vec1.dtype:
                raise TypeError("Vector and scalar dtypes must be the same")
        b = beta.value
    else:
        b = beta

    return _v.plane_rotation(x, y, a, b)


def norm(x, ord=None):
    """
    Returns the vector norm of ``x``, if that is defined.

    The norm returned depends on the ``ord`` parameter, as in SciPy.

    Parameters
    ----------
    ord : {1, 2, inf, 'fro', None}
        Order of the norm.
        inf means NumPy's ``inf`` object.
        'fro' means the string 'fro', and denotes the Frobenius norm.
        If None and self is a Matrix instance, then assumes 'fro'.
    """
    return x.norm(ord)


def prod(A, B):
    """
    Returns ``Mul(A, B)`` where that is defined (see the help for ``Mul``),
    otherwise returns ``(A * B)``.
    """
    if not isinstance(A, MagicMethods):
        return Mul(A, B)
    return (A * B)


def solve(A, B, solver_tag, precond_tag=tags.NoPreconditioner()):
    """
    Solve the linear system expressed by ``A x = B`` for ``x``.

    Parameters
    ----------
    A : (M, M) dense or sparse Matrix
        A square matrix
    B : {Vector, Matrix}
        Right-hand side in ``A x = B``
    solver_tag : SolverTag instance
        Describes the system matrix and solver algorithm.
        See the help for each tag class for more information.
    precond_tag : PreconditionerTag instance

    Returns
    -------
    x : {Vector, Matrix}
        Shape and class of ``x`` matches shape and class of ``B``.

    Raises
    ------
    TypeError
        If ``A`` is not a ``Matrix``  or ``SparseMatrixBase`` instance,
        or ``B`` is neither a ``Matrix`` nor a ``Vector`` instance,
        or if ``tag`` is unsupported.
    """
    if not isinstance(A, MagicMethods):
        raise TypeError("A must be dense or sparse matrix type")
    elif not (issubclass(A.result_container_type, Matrix) or issubclass(A.result_container_type, SparseMatrixBase)):
        raise TypeError("A must be dense or sparse matrix type")

    if precond_tag is None:
        precond_tag = tags.NoPreconditioner()

    if isinstance(solver_tag, tags.SolverWithoutPreconditioner):
        return B.new_instance(solver_tag.vcl_solve_call
                                (A.vcl_leaf, B.vcl_leaf,
                                 solver_tag.vcl_tag))
    else:
        precond_tag.instantiate(A)
        return B.new_instance(solver_tag.vcl_solve_call
                                (A.vcl_leaf, B.vcl_leaf,
                                 solver_tag.vcl_tag,
                                 precond_tag.vcl_precond))
Matrix.solve = solve           # for convenience..
SparseMatrixBase.solve = solve #


def eig(A, tag):
    """
    Solve an eigenvalue problem for matrix ``A``, with results depending
    on ``tag``.

    Parameters
    ----------
    A : Matrix
    tag : eigenvalue computation tag instance
        Must be one of
        * power_iter_tag
        * lanczos_tag
        See the help for each tag class for more information.

    Returns
    -------
    x : {scalar, array-like}
        Return type depends on ``tag``
        * if power_iter_tag, then a scalar of type ``dtype(A).type``
        * if lanczos_tag, then an ``ndarray`` vector with same dtype as ``A``

    Raises
    ------
    TypeError
        If ``A`` is not a ``Matrix`` instance, or if ``tag`` is not understood
    """
    #if not isinstance(A, Matrix):
    #    raise TypeError("A must be a Matrix type")

    if isinstance(tag, tags.PowerIteration):
        return _v.eig(A.vcl_leaf, tag.vcl_tag)
    elif isinstance(tag, tags.Lanczos):
        return _v.eig(A.vcl_leaf, tag.vcl_tag).as_ndarray()
    else:
        raise TypeError("tag must be a supported eigenvalue tag!")
Matrix.eig = eig
SparseMatrixBase.eig = eig


def inplace_qr(A, block_size=16, copy=False):
    """
    TODO docstring
    """
    if copy: A = A.copy()
    vcl_betas = _v.inplace_qr(A.vcl_leaf, block_size)
    betas = Vector(vcl_betas, dtype=A.dtype, context=A.context)
    return A, betas


def recover_q(A, betas):
    """
    TODO docstring
    """
    Q = Matrix(A.shape[0], A.shape[0], 0.0,
               dtype=A.dtype, layout=A.layout, context=A.context)
    R = Matrix(A.shape[0], A.shape[1], 0.0,
               dtype=A.dtype, layout=A.layout, context=A.context)
    _v.recoverQ(A.vcl_leaf, betas.vcl_leaf.as_std_vector(),
                Q.vcl_leaf, R.vcl_leaf)
    return Q, R


def inplace_qr_apply_trans_q(A, betas, b, copy=False):
    """
    TODO docstring
    inplace Q.T.dot(b)
    """
    if copy: b = b.copy()
    _v.inplace_qr_apply_trans_Q(A.vcl_leaf, betas.vcl_leaf.as_std_vector(),
                                b.vcl_leaf)
    return b


def qr(A, block_size=16):
    """
    TODO docstring
    """
    QR, betas = inplace_qr(A, block_size, True)
    Q, R = recover_q(QR, betas)
    return Q, R, betas


def nmf(*args):
    """
    TODO docstring
    args = V, k; V, k, tag; V, W, H; V, W, H, tag
    """
    V = args[0]
    if V.context.domain is not backend.OpenCLMemory:
        raise TypeError("You can only perform NMF with the OpenCL backend")
    if len(args) == 2:
        tag = p.tags.NMF()
        k = args[1]
        W = Matrix(V.shape[0], k, 0.0,
                   dtype=V.dtype, layout=V.layout, context=V.context)
        H = Matrix(k, V.shape[1], 0.0,
                   dtype=V.dtype, layout=V.layout, context=V.context)
    if len(args) == 3:
        if isinstance(args[1], int):
            k = args[1]
            tag = args[2]
            W = Matrix(V.shape[0], k, 0.0,
                       dtype=V.dtype, layout=V.layout, context=V.context)
            H = Matrix(k, V.shape[1], 0.0,
                       dtype=V.dtype, layout=V.layout, context=V.context)
        else:
            tag = p.tags.NMF()
            W = args[1]
            H = args[2]
    else:
        W = args[1]
        H = args[2]
        tag = args[3]
    _v.nmf(V.vcl_leaf, W.vcl_leaf, H.vcl_leaf, tag.vcl_tag)
    return W, H


def fft(input, batch_num=1, sign=-1.0):
    """
    TODO docstring
    1-D: input, output, batch_num=1, sign=-1
    2-D: input, output, sign=-1

    args: input, batch_num, sign
          input [Matrix or Vector]
          input

    return output
    """
    output = input.new_instance()

    if issubclass(input.result_container_type, Vector):
        _v.fft(input.vcl_leaf, output.vcl_leaf, batch_num, sign)
    else:
        _v.fft(input.vcl_leaf, output.vcl_leaf, sign)

    return output

def inplace_fft(input, batch_num=1, sign=-1.0):
    """
    TODO docstring
    1-D: input, batch_num=1, sign=-1
    2-D: input, sign=-1
    return None
    """
    if issubclass(input.result_container_type, Vector):
        _v.inplace_fft(input.vcl_leaf, batch_num, sign)
    else:
        _v.inplace_fft(input.vcl_leaf, sign)

    return input

def ifft(input, batch_num=1, sign=1.0):
    output = fft(input, batch_num, sign)
    _v.normalize(output.vcl_leaf)
    return output

def inplace_ifft(input, batch_num=1, sign=1.0):
    input = inplace_fft(input, batch_num, sign)
    _v.normalize(input.vcl_leaf)
    return input


def convolve(input1, input2):
    """
    TODO docstring
    """
    output = input1.new_instance()
    _v.convolve(input1.vcl_leaf, input2.vcl_leaf, output.vcl_leaf)
    return output


def convolve_i(input1, input2):
    """
    TODO docstring
    """
    output = input1.new_instance()
    _v.convolve_i(input1.vcl_leaf, input2.vcl_leaf, output.vcl_leaf)
    return output

