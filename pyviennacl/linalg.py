from pyviennacl import tags, _viennacl as _v
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
    ord : {1, 2, inf}
        Order of the norm. inf means NumPy's ``inf`` object.
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

    if not isinstance(B, MagicMethods):
        raise TypeError("B must be Matrix or Vector type")
    else:
        result_type = B.result_container_type

    if precond_tag is None:
        precond_tag = tags.NoPreconditioner()

    if isinstance(solver_tag, tags.SolverWithoutPreconditioner):
        return result_type(solver_tag.vcl_solve_call
                           (A.vcl_leaf, B.vcl_leaf,
                            solver_tag.vcl_tag),
                           dtype = B.dtype,
                           layout = B.layout)
    else:
        precond_tag.instantiate(A)
        return result_type(solver_tag.vcl_solve_call
                           (A.vcl_leaf, B.vcl_leaf,
                            solver_tag.vcl_tag,
                            precond_tag.vcl_precond),
                           dtype = B.dtype,
                           layout = B.layout)
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

