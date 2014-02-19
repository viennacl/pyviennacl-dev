#ifndef _PYVIENNACL_DIRECT_SOLVERS_H
#define _PYVIENNACL_DIRECT_SOLVERS_H

#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/matrix.hpp>

#include "viennacl.h"
#include "solve_op_func.hpp"

#define EXPORT_MATRIX_VECTOR_SOLVERS(TYPE, F)                           \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::lower_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::upper_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector<TYPE>,           \
          vcl::matrix_base<TYPE, F>&, vcl::vector_base<TYPE>&,          \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve, 0>);

#define EXPORT_DIRECT_SOLVERS(TYPE)                                     \
  EXPORT_MATRIX_VECTOR_SOLVERS(TYPE, vcl::row_major)                    \
  EXPORT_MATRIX_VECTOR_SOLVERS(TYPE, vcl::column_major)                 \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::lower_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::upper_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::lower_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::upper_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::lower_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::upper_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::lower_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::upper_tag&,                                      \
          op_solve, 0>);                                                \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix<TYPE, vcl::row_major>, \
          vcl::matrix_base<TYPE, vcl::column_major>&,                   \
          vcl::matrix_base<TYPE, vcl::row_major>&,                      \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve, 0>);

#endif
