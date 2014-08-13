from __future__ import division

import math
import numpy as np
import pyviennacl as p
import scipy.sparse.linalg as spspla
from _common import *
from itertools import product

points_x_y = 5

if double_support:
    dtype_tolerances = [('float64', 1.0E-5)]
else:
    dtype_tolerances = [('float32', 1.0E-3)]


matrix_types = [('compressed_matrix', 'p.CompressedMatrix'),
                ('coordinate_matrix', 'p.CoordinateMatrix'),
                ('ell_matrix', 'p.ELLMatrix'),
                ('hyb_matrix', 'p.HybridMatrix')]

rhs_vector_getters = [('vector', 'get_vector')] #,
                      #('vector_range', 'get_vector_range'),
                      #('vector_slice', 'get_vector_slice')]

solvers = [('cg', 'p.tags.CG'),
           #('mixed_precision_cg', 'p.tags.MixedPrecisionCG'),
           ('bicgstab', 'p.tags.BiCGStab'),
           ('gmres', 'p.tags.GMRES')]

preconditioners = [('no_preconditioner', 'p.tags.NoPreconditioner',
                    ['compressed_matrix', 'coordinate_matrix',
                     'ell_matrix', 'hyb_matrix']),
                   ('ichol0', 'p.tags.ICHOL0', ['compressed_matrix']),
                   ('ilu0', 'p.tags.ILU0', ['compressed_matrix']),
                   ('ilut', 'p.tags.ILUT', ['compressed_matrix']),
                   ('block_ilu0', 'p.tags.BlockILU0', ['compressed_matrix']),
                   ('block_ilut', 'p.tags.BlockILUT', ['compressed_matrix']),
                   ('jacobi', 'p.tags.Jacobi', ['compressed_matrix']),
                   ('row_scaling', 'p.tags.RowScaling', ['compressed_matrix', 'coordinate_matrix']),
                   #('amg', 'p.tags.AMG', ['compressed_matrix']),
                   ('spai', 'p.tags.SPAI', ['compressed_matrix']),
                   ('fspai', 'p.tags.FSPAI', ['compressed_matrix'])]


for d_t_, solver_, sparse_type_, vector_getter_, precond_ in product(dtype_tolerances, solvers, matrix_types, rhs_vector_getters, preconditioners):
    dt_ = d_t_[0]
    tol_ = d_t_[1]
    solver_name = solver_[0]
    solver_tag_name = solver_[1]
    if sparse_type_[0] not in precond_[2]: continue
    precond_tag_name = precond_[1]

    def A_solve_b_test_factory(dt, tol, sparse_type, vector_getter, solver_tag_type, precond_tag_type):
        def _test():
            solver_tag = solver_tag_type(tolerance=tol/10)
            precond_tag = precond_tag_type()

            vcl_system = sparse_type.generate_fdm_laplace(points_x_y, points_x_y, dtype=dt)
            #vcl_system = get_sparse_matrix(10, dtype=dt, sparse_type=sparse_type)
            numpy_system = vcl_system.as_ndarray() # TODO: SciPy-ise

            numpy_solution, vcl_solution = vector_getter(vcl_system.size1, dt, vector=np.ones(vcl_system.size1).astype(dt))

            numpy_rhs, vcl_rhs = vector_getter(vcl_system.size1, dt, vector=vcl_system*vcl_solution)

            # solve using pyviennacl
            vcl_solution = p.solve(vcl_system, vcl_rhs, solver_tag, precond_tag)

            # compare with known solution
            act_diff = math.fabs(diff(vcl_rhs, vcl_system.dot(vcl_solution)))
            assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
            del solver_tag, precond_tag, vcl_system, vcl_solution, vcl_rhs

        return _test
        
    exec("test_%s_with_%s_solve_%s_A_%s_b_%s = A_solve_b_test_factory(p.%s, %g, %s, %s, %s, %s)" % (solver_name, precond_[0], sparse_type_[0], vector_getter_[0], dt_, dt_, tol_, sparse_type_[1], vector_getter_[1], solver_tag_name, precond_tag_name))

