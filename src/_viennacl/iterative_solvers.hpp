#ifndef _PYVIENNACL_ITERATIVE_SOLVERS_H
#define _PYVIENNACL_ITERATIVE_SOLVERS_H

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/gmres.hpp>

#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>

#include "viennacl.h"
#include "solve_op_func.hpp"

#define EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, F)                       \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::matrix_base<TYPE, F>&, vcl::vector<TYPE>&,               \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);


#define EXPORT_DENSE_ITERATIVE_SOLVERS(TYPE)                    \
  EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, vcl::row_major);       \
  EXPORT_DENSE_ITERATIVE_SOLVERS_F(TYPE, vcl::column_major); 

#define EXPORT_SPARSE_ITERATIVE_SOLVERS(TYPE)                           \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,            \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::ell_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::ell_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::ell_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::hyb_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::cg_tag&,                                         \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::hyb_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::bicgstab_tag&,                                   \
          op_solve, 0>);                                                \
  bp::def("iterative_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,        \
          vcl::hyb_matrix<TYPE>&, vcl::vector<TYPE>&,                   \
          vcl::linalg::gmres_tag&,                                      \
          op_solve, 0>);


#endif
