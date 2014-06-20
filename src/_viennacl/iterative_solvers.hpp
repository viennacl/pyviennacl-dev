#ifndef _PYVIENNACL_ITERATIVE_SOLVERS_HPP
#define _PYVIENNACL_ITERATIVE_SOLVERS_HPP

#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>

#ifdef VIENNACL_WITH_OPENCL
#include <viennacl/linalg/mixed_precision_cg.hpp>
#endif

#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>

#include "solve_op_func.hpp"
#include "preconditioners.hpp"

// TODO:: mixed_precision_cg supports only OpenCL, compressed_matrix, and no preconditioners
#ifdef VIENNACL_WITH_OPENCL

#define EXPORT_OPENCL_PRECONDITIONED_SOLVERS(MAT, TYPE)                 \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::amg_precond<MAT>); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::spai_precond<MAT>); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::fspai_precond<MAT>);

#define EXPORT_OPENCL_NON_PRECONDITIONED_SOLVERS(MAT, TYPE)             \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::mixed_precision_cg_tag&, vcl::linalg::no_precond, \
          op_solve_precond>);

#define EXPORT_OPENCL_SOLVERS(MAT, TYPE)                \
  EXPORT_OPENCL_PRECONDITIONED_SOLVERS(MAT, TYPE);      \
  EXPORT_OPENCL_NON_PRECONDITIONED_SOLVERS(MAT, TYPE);

#else

#define EXPORT_OPENCL_PRECONDITIONED_SOLVERS(MAT, TYPE)
#define EXPORT_OPENCL_NON_PRECONDITIONED_SOLVERS(MAT, TYPE)
#define EXPORT_OPENCL_SOLVERS(MAT, TYPE)

#endif // VIENNACL_WITH_OPENCL

#define EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, PRECOND)            \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::cg_tag&, PRECOND&,                               \
          op_solve_precond>);                                           \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::bicgstab_tag&, PRECOND&,                         \
          op_solve_precond>);                                           \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::gmres_tag&, PRECOND&,                            \
          op_solve_precond>);

#define EXPORT_ALL_ITERATIVE_SOLVERS(MAT, TYPE)                         \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::no_precond); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::ichol0_precond<MAT >); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::ilut_precond<MAT >); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::ilu0_precond<MAT >); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE,                           \
                                   vcl::linalg::block_ilu_precond<MAT COMMA vcl::linalg::ilut_tag>); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE,                           \
                                   vcl::linalg::block_ilu_precond<MAT COMMA vcl::linalg::ilu0_tag>); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::jacobi_precond<MAT>); \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, vcl::linalg::row_scaling<MAT>); \
  EXPORT_OPENCL_SOLVERS(MAT, TYPE);

// TODO: Other sparse types than compressed don't support many preconditioners
#define EXPORT_SPARSE_ITERATIVE_SOLVERS(TYPE)                           \
  EXPORT_ALL_ITERATIVE_SOLVERS(vcl::compressed_matrix<TYPE>, TYPE)      \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>,        \
                                   TYPE, vcl::linalg::no_precond);      \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>,        \
                                   TYPE,                                \
                                   vcl::linalg::jacobi_precond<         \
                                   vcl::coordinate_matrix<TYPE> >);     \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>,        \
                                   TYPE,                                \
                                   vcl::linalg::row_scaling<            \
                                   vcl::coordinate_matrix<TYPE> >);     \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(vcl::ell_matrix<TYPE>,               \
                                   TYPE, vcl::linalg::no_precond);      \
  EXPORT_ITERATIVE_SOLVERS_GENERAL(vcl::hyb_matrix<TYPE>,               \
                                     TYPE, vcl::linalg::no_precond);

#endif
