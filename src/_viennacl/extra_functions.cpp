#include "viennacl.h"

#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/norm_frobenius.hpp>
#include <viennacl/fft.hpp>

DO_OP_FUNC(op_inner_prod)
{
  return vcl::linalg::inner_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_outer_prod)
{
  return vcl::linalg::outer_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_pow)
{
  return vcl::linalg::element_pow(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_norm_1)
{
  return vcl::linalg::norm_1(o.operand1);
} };

DO_OP_FUNC(op_norm_2)
{
  return vcl::linalg::norm_2(o.operand1);
} };

DO_OP_FUNC(op_norm_inf)
{
  return vcl::linalg::norm_inf(o.operand1);
} };

DO_OP_FUNC(op_norm_frobenius)
{
  return vcl::linalg::norm_frobenius(o.operand1);
} };

DO_OP_FUNC(op_plane_rotation)
{
  vcl::linalg::plane_rotation(o.operand1, o.operand2,
			      o.operand3, o.operand4);
  return bp::object();
} };

DO_OP_FUNC(op_fft)
{
  vcl::fft(o.operand1, o.operand2);
  return bp::object();
} };

DO_OP_FUNC(op_inplace_fft)
{
  vcl::inplace_fft(o.operand1);
  return bp::object();
} };

DO_OP_FUNC(op_ifft)
{
  vcl::ifft(o.operand1, o.operand2);
  return bp::object();
} };

DO_OP_FUNC(op_inplace_ifft)
{
  vcl::inplace_ifft(o.operand1);
  return bp::object();
} };

#define EXPORT_FUNCTIONS_F(TYPE, F)                                     \
  bp::def("outer", pyvcl_do_2ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_outer_prod, 0>);                                           \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,         \
          vcl::matrix_base<TYPE, F>&, vcl::matrix_base<TYPE, F>&,       \
          op_element_pow, 0>);                                          \
  bp::def("norm_frobenius", pyvcl_do_1ary_op<vcl::scalar<TYPE>,         \
          vcl::matrix<TYPE, F>&,                                        \
          op_norm_frobenius, 0>);

#define EXPORT_FUNCTIONS(TYPE)                                          \
  EXPORT_FUNCTIONS_F(TYPE, vcl::row_major);                             \
  EXPORT_FUNCTIONS_F(TYPE, vcl::column_major);                          \
  bp::def("inner_prod", pyvcl_do_2ary_op<vcl::scalar<TYPE>,             \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_inner_prod, 0>);                                           \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::vector<TYPE>,            \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_element_pow, 0>);                                          \
  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,                \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          TYPE, TYPE,                                                   \
          op_plane_rotation, 0>);                                       \
  bp::def("norm_1", pyvcl_do_1ary_op<vcl::scalar<TYPE>,                 \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_1, 0>);                                               \
  bp::def("norm_2", pyvcl_do_1ary_op<vcl::scalar<TYPE>,                 \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_2, 0>);                                               \
  bp::def("norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,               \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_inf, 0>);                                             \
  bp::def("fft", pyvcl_do_2ary_op<bp::object,                           \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&,                       \
          op_fft, 0>);                                                  \
  bp::def("ifft", pyvcl_do_2ary_op<bp::object,                          \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&,                       \
          op_ifft, 0>);                                                 \
  bp::def("inplace_fft", pyvcl_do_1ary_op<bp::object,                   \
          vcl::vector<TYPE>&,                                           \
          op_inplace_fft, 0>);                                          \
  bp::def("inplace_ifft", pyvcl_do_1ary_op<bp::object,                  \
          vcl::vector<TYPE>&,                                           \
          op_inplace_ifft, 0>);                                                 


PYVCL_SUBMODULE(extra_functions)
{
  /* TODO missing: char, short, uchar, ushort
     Some of these only make compile on Windows for float types -- eg norm_2, which
       tries to do a sqrt on a long without converting it to some float type.
  EXPORT_FUNCTIONS(int);
  EXPORT_FUNCTIONS(long);
  EXPORT_FUNCTIONS(uint);
  EXPORT_FUNCTIONS(ulong);
  */
  EXPORT_FUNCTIONS(double);
  EXPORT_FUNCTIONS(float);

}
