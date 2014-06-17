#ifndef _PYVIENNACL_ITERATIVE_SOLVERS_H
#define _PYVIENNACL_ITERATIVE_SOLVERS_H

#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/mixed_precision_cg.hpp>

#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>

#include "viennacl.h"
#include "solve_op_func.hpp"
#include "preconditioners.hpp"

#define EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, F, PRECOND)              \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::cg_tag&, PRECOND&,                               \
          op_solve_precond, 0>);                                        \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::mixed_precision_cg_tag&, PRECOND&,               \
          op_solve_precond, 0>);                                        \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::bicgstab_tag&, PRECOND&,                         \
          op_solve_precond, 0>);                                        \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::gmres_tag&, PRECOND&,                            \
          op_solve_precond, 0>);

#define EXPORT_DENSE_ITERATIVE_SOLVERS(TYPE, PRECOND)                   \
  EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, vcl::row_major, PRECOND);      \
  EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, vcl::column_major, PRECOND); 


#define EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(MAT, TYPE, PRECOND)     \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::cg_tag&, PRECOND&,                               \
          op_solve_precond, 0>);                                        \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::bicgstab_tag&, PRECOND&,                         \
          op_solve_precond, 0>);                                        \
  bp::def("iterative_solve", pyvcl_do_4ary_op<vcl::vector<TYPE>,        \
          MAT&, vcl::vector<TYPE>&,                                     \
          vcl::linalg::gmres_tag&, PRECOND&,                            \
          op_solve_precond, 0>);

#define COMMA ,
#define EXPORT_SPARSE_ITERATIVE_SOLVERS(TYPE)                           \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE, vcl::linalg::no_precond); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::ichol0_precond<  \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::ilut_precond<    \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::ilu0_precond<    \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::block_ilu_precond<vcl::compressed_matrix<TYPE> COMMA vcl::linalg::ilut_tag>); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::block_ilu_precond<vcl::compressed_matrix<TYPE> COMMA vcl::linalg::ilu0_tag>); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::jacobi_precond<  \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::row_scaling<     \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::amg_precond<     \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::spai_precond<    \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::compressed_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::fspai_precond<   \
                                          vcl::compressed_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>, \
                                          TYPE, vcl::linalg::no_precond); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::jacobi_precond<  \
                                          vcl::coordinate_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::coordinate_matrix<TYPE>, \
                                          TYPE,                         \
                                          vcl::linalg::row_scaling<     \
                                          vcl::coordinate_matrix<TYPE> >); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::ell_matrix<TYPE>,        \
                                          TYPE, vcl::linalg::no_precond); \
  EXPORT_SPARSE_ITERATIVE_SOLVERS_GENERAL(vcl::hyb_matrix<TYPE>,        \
                                          TYPE, vcl::linalg::no_precond);

#endif
