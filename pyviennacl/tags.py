"""
TODO docstring
"""

from pyviennacl import _viennacl as _v, backend, vcl_statement_node_subtype_strings as vcl_type_strings, vcl_statement_node_numeric_type_strings as dtype_strings


class PreconditionerTag(object):
    vcl_precond_type = None
    vcl_tag_type = None

    def instantiate(self, leaf):
        self.vcl_precond_type = getattr(_v, self.vcl_precond_type_name + "_" + type(leaf.vcl_leaf).__name__)
        vcl_precond = self.vcl_precond_type(leaf.vcl_leaf, self.vcl_tag)
        self.vcl_precond = vcl_precond


class SolverTag(object): pass

class SolverWithoutPreconditioner(object): pass

class EigenvalueTag(object): pass


class NoPreconditioner(PreconditionerTag):
    vcl_tag_type = None
    vcl_precond_type = _v.no_precond

    def __init__(self):
        self.vcl_tag = None
        self.vcl_precond = self.vcl_precond_type()

    def instantiate(self, leaf): pass


class ICHOL0(PreconditionerTag):
    vcl_tag_type = _v.ichol0_tag
    vcl_precond_type_name = 'ichol0_precond'

    def __init__(self):
        self.vcl_tag = self.vcl_tag_type()


class ILUT(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.ilut_tag
    vcl_precond_type_name = 'ilut_precond'

    def __init__(self, entries_per_row=20, drop_tolerance=1e-4, with_level_scheduling=False):
        """
        TODO docstring
        """
        self.vcl_tag = self.vcl_tag_type(entries_per_row, drop_tolerance, with_level_scheduling)

    @property
    def entries_per_row(self):
        return self.vcl_tag.entries_per_row

    @property
    def drop_tolerance(self):
        return self.vcl_tag.drop_tolerance

    @property
    def with_level_scheduling(self):
        return self.vcl_tag.with_level_scheduling


class BlockILUT(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.ilut_tag
    vcl_precond_type_name = 'block_ilut_precond'

    def __init__(self, entries_per_row=20, drop_tolerance=1e-4, with_level_scheduling=False, num_blocks=8):
        """
        TODO docstring
        """
        self.num_blocks = num_blocks
        self.vcl_tag = self.vcl_tag_type(entries_per_row, drop_tolerance, with_level_scheduling)

    def instantiate(self, leaf):
        self.vcl_precond_type = getattr(_v, self.vcl_precond_type_name + "_" + type(leaf.vcl_leaf).__name__)
        vcl_precond = self.vcl_precond_type(leaf.vcl_leaf, self.vcl_tag, self.num_blocks)
        self.vcl_precond = vcl_precond

    @property
    def entries_per_row(self):
        return self.vcl_tag.entries_per_row

    @property
    def drop_tolerance(self):
        return self.vcl_tag.drop_tolerance

    @property
    def with_level_scheduling(self):
        return self.vcl_tag.with_level_scheduling


class ILU0(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.ilu0_tag
    vcl_precond_type_name = 'ilu0_precond'

    def __init__(self, with_level_scheduling=False):
        """
        TODO docstring
        """
        self.vcl_tag = self.vcl_tag_type(with_level_scheduling)

    @property
    def with_level_scheduling(self):
        return self.vcl_tag.with_level_scheduling


class BlockILU0(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.ilu0_tag
    vcl_precond_type_name = 'block_ilu0_precond'

    def __init__(self, with_level_scheduling=False, num_blocks=8):
        """
        TODO docstring
        """
        self.num_blocks = num_blocks
        self.vcl_tag = self.vcl_tag_type(with_level_scheduling)

    def instantiate(self, leaf):
        self.vcl_precond_type = getattr(_v, self.vcl_precond_type_name + "_" + type(leaf.vcl_leaf).__name__)
        vcl_precond = self.vcl_precond_type(leaf.vcl_leaf, self.vcl_tag, self.num_blocks)
        self.vcl_precond = vcl_precond

    @property
    def with_level_scheduling(self):
        return self.vcl_tag.with_level_scheduling


class Jacobi(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.jacobi_tag
    vcl_precond_type_name = 'jacobi_precond'

    def __init__(self):
        """
        TODO docstring
        """
        self.vcl_tag = self.vcl_tag_type()


class RowScaling(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.row_scaling_tag
    vcl_precond_type_name = 'row_scaling_precond'

    def __init__(self, p=2):
        """
        TODO docstring (nb - norm is l^p norm)
        """
        self.vcl_tag = self.vcl_tag_type(p)

    @property
    def norm(self):
        return self.vcl_tag.norm


class AMG(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.amg_tag
    vcl_precond_type_name = 'amg_precond'

    def __init__(self, coarse=1,
                 interpol=1,
                 threshold=0.25,
                 interpol_weight=0.2,
                 jacobi_weight=1.0,
                 presmooth=1,
                 postsmooth=1,
                 coarse_levels=0):
        """
        TODO docstring
        """
        if not backend.WITH_OPENCL:
            raise NotImplementedError("AMG preconditioner only available with OpenCL")
        self.vcl_tag = self.vcl_tag_type(coarse, interpol, threshold, interpol_weight, jacobi_weight, presmooth, postsmooth, coarse_levels)

    @property
    def coarse(self):
        return self.vcl_tag.coarse

    @property
    def interpol(self):
        return self.vcl_tag.interpol

    @property
    def threshold(self):
        return self.vcl_tag.threshold

    @property
    def interpol_weight(self):
        return self.vcl_tag.interpol_weight

    @property
    def jacobi_weight(self):
        return self.vcl_tag.jacobi_weight

    @property
    def presmooth(self):
        return self.vcl_tag.presmooth

    @property
    def postsmooth(self):
        return self.vcl_tag.postsmooth

    @property
    def coarse_levels(self):
        return self.vcl_tag.coarse_levels


class SPAI(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.spai_tag
    vcl_precond_type_name = 'spai_precond'

    def __init__(self, residual_norm_threshold=1e-3,
                 iteration_limit=5,
                 residual_threshold=1e-2,
                 is_static=False,
                 is_right=False):
        if not backend.WITH_OPENCL:
            raise NotImplementedError("SPAI preconditioner only available with OpenCL")
        self.vcl_tag = self.vcl_tag_type(residual_norm_threshold, iteration_limit, residual_threshold, is_static, is_right)

    @property
    def residual_norm_threshold(self):
        return self.vcl_tag.residual_norm_threshold

    @property
    def iteration_limit(self):
        return self.vcl_tag.iteration_limit

    @property
    def residual_threshold(self):
        return self.vcl_tag.residual_threshold

    @property
    def is_static(self):
        return self.vcl_tag.is_static

    @property
    def is_right(self):
        return self.vcl_tag.is_right


class FSPAI(PreconditionerTag):
    """
    TODO docstring
    """
    vcl_tag_type = _v.fspai_tag
    vcl_precond_type_name = 'fspai_precond'

    def __init__(self, residual_norm_threshold=1e-3,
                 iteration_limit=5,
                 residual_threshold=1e-2,
                 is_static=False,
                 is_right=False):
        if not backend.WITH_OPENCL:
            raise NotImplementedError("FSPAI preconditioner only available with OpenCL")
        self.vcl_tag = self.vcl_tag_type(residual_norm_threshold, iteration_limit, residual_threshold, is_static, is_right)

    @property
    def residual_norm_threshold(self):
        return self.vcl_tag.residual_norm_threshold

    @property
    def iteration_limit(self):
        return self.vcl_tag.iteration_limit

    @property
    def residual_threshold(self):
        return self.vcl_tag.residual_threshold

    @property
    def is_static(self):
        return self.vcl_tag.is_static

    @property
    def is_right(self):
        return self.vcl_tag.is_right


class Lower(SolverTag, SolverWithoutPreconditioner):
    """
    Instruct the solver to solve for a lower triangular system matrix
    """
    vcl_tag = _v.lower_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)


class UnitLower(SolverTag, SolverWithoutPreconditioner):
    """
    Instruct the solver to solve for a unit lower triangular system matrix
    """
    vcl_tag = _v.unit_lower_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)


class Upper(SolverTag, SolverWithoutPreconditioner):
    """
    Instruct the solver to solve for an upper triangular system matrix
    """
    vcl_tag = _v.upper_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)


class UnitUpper(SolverTag, SolverWithoutPreconditioner):
    """
    Instruct the solver to solve for a unit upper triangular system matrix
    """
    vcl_tag = _v.unit_upper_tag()
    def vcl_solve_call(self, *args):
        return _v.direct_solve(*args)


class CG(SolverTag):
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


class BiCGStab(SolverTag):
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


class GMRES(SolverTag):
    """
    Instruct the solver to solve using the GMRES solver.

    Used for supplying solver parameters.
    """
    def vcl_solve_call(self, *args):
        return _v.iterative_solve(*args)

    def __init__(self,tolerance = 1e-8, max_iterations = 300, krylov_dim = 20):
        """
        TODO docstring

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


class PowerIteration(EigenvalueTag):
    """
    Instruct the eigenvalue computation to use the power iteration algorithm.

    Used for supplying eigenvalue computation parameters.
    """
    def __init__(self, factor = 1e-8, max_iterations = 50000):
        """
        TODO docstring

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


class Lanczos(EigenvalueTag):
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

