#include "viennacl.h"

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
          op_norm_inf, 0>);


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
