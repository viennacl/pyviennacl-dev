#ifndef _PYVIENNACL_DIRECT_SOLVERS_HPP
#define _PYVIENNACL_DIRECT_SOLVERS_HPP

#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/sparse_matrix_operations.hpp>
#include <viennacl/matrix.hpp>

#include "sparse_matrix.hpp"
#include "solve_op_func.hpp"

#define EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(MAT, TYPE)          \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&, vcl::vector_base<TYPE>&,                                \
          vcl::linalg::lower_tag&,                                      \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&, vcl::vector_base<TYPE>&,                                \
          vcl::linalg::unit_lower_tag&,                                 \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&, vcl::vector_base<TYPE>&,                                \
          vcl::linalg::upper_tag&,                                      \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&, vcl::vector_base<TYPE>&,                                \
          vcl::linalg::unit_upper_tag&,                                 \
          op_inplace_solve>);

#define EXPORT_DIRECT_INPLACE_SOLVERS(MAT, TYPE)                        \
  EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(MAT, TYPE)                \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&,                                                         \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::lower_tag&,                                      \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&,                                                         \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::unit_lower_tag&,                                 \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&,                                                         \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::upper_tag&,                                      \
          op_inplace_solve>);                                           \
  bp::def("direct_inplace_solve", pyvcl_do_3ary_op<bp::object,          \
          MAT&,                                                         \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::unit_upper_tag&,                                 \
          op_inplace_solve>);

#define EXPORT_DENSE_MATRIX_VECTOR_DIRECT_SOLVERS(TYPE)                 \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector_base<TYPE>,      \
          vcl::matrix_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          vcl::linalg::lower_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector_base<TYPE>,      \
          vcl::matrix_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector_base<TYPE>,      \
          vcl::matrix_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          vcl::linalg::upper_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::vector_base<TYPE>,      \
          vcl::matrix_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve>);

#define EXPORT_DENSE_DIRECT_SOLVERS(TYPE)                               \
  EXPORT_DIRECT_INPLACE_SOLVERS(vcl::matrix_base<TYPE>, TYPE)           \
  EXPORT_DENSE_MATRIX_VECTOR_DIRECT_SOLVERS(TYPE)                       \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix_base<TYPE>,      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::lower_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix_base<TYPE>,      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::unit_lower_tag&,                                 \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix_base<TYPE>,      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::upper_tag&,                                      \
          op_solve>);                                                   \
  bp::def("direct_solve", pyvcl_do_3ary_op<vcl::matrix_base<TYPE>,      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::matrix_base<TYPE>&,                                      \
          vcl::linalg::unit_upper_tag&,                                 \
          op_solve>);

#endif
