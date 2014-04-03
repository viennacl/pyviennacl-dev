from pyviennacl import _viennacl as _v
from pyviennacl.pycore import (Matrix, SparseMatrixBase, ScalarBase, Vector,
                               Node, MagicMethods, Mul,
                               vcl_statement_node_numeric_type_strings)
from numpy import (ndarray, array, dtype,
                   result_type as np_result_type)
import logging

log = logging.getLogger(__name__)

class no_precond:
    vcl_precond = _v.no_precond()

class precond_tag:
    pass

class lower_tag:
    """
    Instruct the solver to solve for a lower triangular system matrix
    """
    vcl_tag = _v.lower_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)

class unit_lower_tag:
    """
    Instruct the solver to solve for a unit lower triangular system matrix
    """
    vcl_tag = _v.unit_lower_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)

class upper_tag:
    """
    Instruct the solver to solve for an upper triangular system matrix
    """
    vcl_tag = _v.upper_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)

class unit_upper_tag:
    """
    Instruct the solver to solve for a unit upper triangular system matrix
    """
    vcl_tag = _v.unit_upper_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)

class cg_tag(precond_tag):
    """
    Instruct the solver to solve using the conjugate gradient solver.

    Assumes that the system matrix is symmetric positive definite.

    Used for supplying solver parameters.
    """
    def vcl_solve_call(self, *args):
        return _v.iterative_solve(*args)

    def __init__(self, tolerance = 1e-8, max_iterations = 300):
        """
        Construct a cg_tag.

        Parameters
        ----------
        tolerance : float, optional
            Relative tolerance for the residual
            (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
        max_iterations : int, optional
            The maximum number of iterations
        """
        self.vcl_tag = _v.cg_tag(tolerance, max_iterations)

    @property
    def tolerance(self):
        """
        The relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        The maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def iters(self):
        """
        The number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        The estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class bicgstab_tag(precond_tag):
    """
    Instruct the solver to solve using the stabilised bi-conjugate gradient
    (BiCGStab) solver.

    Assumes that the system matrix is non-symmetric.

    Used for supplying solver parameters.
    """
    def vcl_solve_call(self, *args):
        return _v.iterative_solve(*args)

    def __init__(self, tolerance = 1e-8, 
                 max_iterations = 400, max_iterations_before_restart = 200):
        """
        Construct a bicgstab_tag.

        Parameters
        ----------
        tolerance : float, optional
            Relative tolerance for the residual
            (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
        max_iterations : int, optional
            Maximum number of iterations
        max_iterations_before restart : int, optional
            Maximum number of iterations before BiCGStab is reinitialised,
            to avoid accumulation of round-off errors.
        """
        self.vcl_tag = _v.bicgstab_tag(tolerance, max_iterations,
                                       max_iterations_before_restart)

    @property
    def tolerance(self):
        """
        The relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        The maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def max_iterations(self):
        """
        The maximum number of iterations before a restart
        """
        return self.vcl_tag.max_iterations_before_restart

    @property
    def iters(self):
        """
        The number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        The estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class gmres_tag(precond_tag):
    """
    Instruct the solver to solve using the GMRES solver.

    Used for supplying solver parameters.
    """
    def vcl_solve_call(self, *args):
        return _v.iterative_solve(*args)

    def __init__(self,tolerance = 1e-8, max_iterations = 300, krylov_dim = 20):
        """
        Construct a gmres_tag
        
        Parameters
        ----------
        tolerance : float, optional
            Relative tolerance for the residual
            (solver quits if ||r|| < tolerance * ||r_initial|| obtains)
        max_iterations : int, optional
            Maximum number of iterations, including restarts
        krylov_dim : int, optional
            The maximum dimension of the Krylov space before restart
            (number of restarts can be computed as max_iterations / krylov_dim)
        """
        self.vcl_tag = _v.gmres_tag(tolerance, max_iterations, krylov_dim)

    @property
    def tolerance(self):
        """
        The relative tolerance
        """
        return self.vcl_tag.tolerance

    @property
    def max_iterations(self):
        """
        The maximum number of iterations
        """
        return self.vcl_tag.max_iterations

    @property
    def krylov_dim(self):
        """
        The maximum dimension of the Krylov space before restart
        """
        return self.vcl_tag.krylov_dim

    @property
    def max_restarts(self):
        """
        The maximum number of GMRES restarts
        """
        return self.vcl_tag.max_restarts

    @property
    def iters(self):
        """
        The number of solver iterations
        """
        return self.vcl_tag.iters

    @property
    def error(self):
        """
        The estimated relative error at the end of the solver run
        """
        return self.vcl_tag.error


class power_iter_tag:
    """
    Instruct the eigenvalue computation to use the power iteration algorithm.

    Used for supplying eigenvalue computation parameters.
    """
    def __init__(self, factor = 1e-8, max_iterations = 50000):
        """
        Construct a power_iter_tag.
        
        Parameters
        ----------
        factor : float, optional
            Halt when the eigenvalue does not change more than this value.
        max_iterations : int, optional
            Maximum number of iterations to compute.
        """
        self.vcl_tag = _v.power_iter_tag(factor, max_iterations)

    @property
    def factor(self):
        """
        The termination factor.

        If the eigenvalue does not change more than this value, the algorithm
        stops.
        """
        return self.vcl_tag.factor

    @property
    def max_iterations(self):
        """
        The maximum number of iterations
        """
        return self.vcl_tag.max_iterations


class lanczos_tag:
    """
    Instruct the eigenvalue computation to use the Lanczos algorithm.

    Used for supplying eigenvalue computation parameters.
    """
    def __init__(self, factor = 0.75, num_eig = 10, method = 0, krylov = 100):
        """
        Construct a lanczos_tag.

        Parameters
        ----------
        factor : float 
            Exponent of epsilon (reorthogonalisation batch tolerance)
        num_eig : int 
            Number of eigenvalues to return
        method : {0, 1, 2}
            0 for partial reorthogonalisation
            1 for full reorthogonalisation
            2 for Lanczos without reorthogonalisation
        krylov : int
            Maximum Krylov-space size
        """
        self.vcl_tag = _v.lanczos_tag(factor, num_eig, method, krylov)

    @property
    def factor(self):
        """
        The tolerance factor for reorthogonalisation batches, expressed as
        the exponent of epsilon.
        """
        return self.vcl_tag.factor

    @property
    def num_eigenvalues(self):
        """
        The number of eigenvalues to return.
        """
        return self.vcl_tag.num_eigenvalues

    @property
    def krylov_size(self):
        """
        The size of the Kylov space.
        """
        return self.vcl_tag.krylov_size

    @property
    def method(self):
        """
        The reorthogonalisation method choice.
        """
        return self.vcl_tag.method


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


def solve(A, B, tag, precond = None):
    """
    Solve the linear system expressed by ``A x = B`` for ``x``.

    Parameters
    ----------
    A : (M, M) dense or sparse Matrix
        A square matrix
    B : {Vector, Matrix}
        Right-hand side in ``A x = B``
    tag : solver tag instance
        Describes the system matrix and solver algorithm.
        Must be one of:
        * upper_tag
        * unit_upper_tag
        * lower_tag
        * unit_lower_tag
        * cg_tag
        * bicgstab_tag
        * gmres_tag
        See the help for each tag class for more information.

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
    if not (isinstance(A, Matrix) or isinstance(A, SparseMatrixBase)):
        raise TypeError("A must be dense or sparse matrix type")

    if isinstance(B, Matrix):
        result_type = Matrix
    elif isinstance(B, Vector):
        result_type = Vector
        if not isinstance(B.vcl_leaf,
                getattr(_v, "vector_" + vcl_statement_node_numeric_type_strings[
                    B.statement_node_numeric_type])):
            B = Vector(B)
    else:
        raise TypeError("B must be Matrix or Vector type")

    try:
        if isinstance(tag, precond_tag) and not (precond is None):
            return result_type(tag.vcl_solve_call(A.vcl_leaf, B.vcl_leaf,
                                                  tag.vcl_tag, precond.vcl_precond),
                               dtype = B.dtype,
                               layout = B.layout)
        else:
            return result_type(tag.vcl_solve_call(A.vcl_leaf, B.vcl_leaf, tag.vcl_tag),
                               dtype = B.dtype,
                               layout = B.layout)
    except AttributeError:
        raise TypeError("tag must be a supported solver tag!")
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

    if isinstance(tag, power_iter_tag):
        return _v.eig(A.vcl_leaf, tag.vcl_tag)
    elif isinstance(tag, lanczos_tag):
        return _v.eig(A.vcl_leaf, tag.vcl_tag).as_ndarray()
    else:
        raise TypeError("tag must be a supported eigenvalue tag!")
Matrix.eig = eig
SparseMatrixBase.eig = eig

def ilu(A, config):
    return NotImplemented


## And QR decomposition..?

