#ifndef _PYVIENNACL_DIRECT_SOLVERS_HPP
#define _PYVIENNACL_DIRECT_SOLVERS_HPP

#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/matrix.hpp>

#include "solve_op_func.hpp"

#define EXPORT_MATRIX_VECTOR_SOLVERS(TYPE, F)                           \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::lower_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::upper_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve>);                                                   \


#define EXPORT_DIRECT_SOLVERS_F(TYPE, F)                                  \
  EXPORT_MATRIX_VECTOR_SOLVERS(TYPE, F)                                 \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>, \
          vcl::matrix_base<TYPE, F>&,                                     \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::linalg::lower_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>, \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>, \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::linalg::upper_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, F>, \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::matrix_base<TYPE, F>&,                                      \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve>);

#define EXPORT_DIRECT_SOLVERS(TYPE)                     \
  EXPORT_DIRECT_SOLVERS_F(TYPE, vcl::row_major)         \
  EXPORT_DIRECT_SOLVERS_F(TYPE, vcl::column_major)

#endif
