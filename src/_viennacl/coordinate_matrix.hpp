#ifndef _PYVIENNACL_COORDINATE_MATRIX_HPP
#define _PYVIENNACL_COORDINATE_MATRIX_HPP

#include "sparse_matrix.hpp"

// TODO: coordinate matrix does not have [switch_]memory_context yet..

#define EXPORT_COORDINATE_MATRIX(TYPE)                                  \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::coordinate_matrix<TYPE>,         \
                                  const vcl::coordinate_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle,                               \
                                  get_coordinate_matrix_##TYPE##_handle, \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::coordinate_matrix<TYPE>,         \
                                  const vcl::coordinate_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle12,                              \
                                  get_coordinate_matrix_##TYPE##_handle12, \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::coordinate_matrix<TYPE>,         \
                                  const vcl::coordinate_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle3,                              \
                                  get_coordinate_matrix_##TYPE##_handle3, \
                                  () const);                            \
  bp::class_<vcl::coordinate_matrix<TYPE>,                              \
             vcl::tools::shared_ptr<vcl::coordinate_matrix<TYPE> >,     \
             boost::noncopyable >                                       \
  ("coordinate_matrix_" #TYPE, bp::no_init)                             \
  /*.add_property("memory_domain",                                      \
                &vcl::coordinate_matrix<TYPE>::memory_context,          \
                &vcl::coordinate_matrix<TYPE>::switch_memory_context) */ \
  .add_property("handle", bp::make_function                             \
                (get_coordinate_matrix_##TYPE##_handle,                 \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle12", bp::make_function                           \
                (get_coordinate_matrix_##TYPE##_handle12,               \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle3", bp::make_function                            \
                (get_coordinate_matrix_##TYPE##_handle3,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("groups",                                               \
                make_function(&vcl::coordinate_matrix<TYPE>::groups,    \
                              bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size1",                                                \
                make_function(&vcl::coordinate_matrix<TYPE>::size1,     \
                              bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::coordinate_matrix<TYPE>::size2,     \
                              bp::return_value_policy<bp::return_by_value>())) \
  .add_property("nnz",                                                  \
                make_function(&vcl::coordinate_matrix<TYPE>::nnz,       \
                              bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,               \
       op_prod>)                                                        \
  ;
    /* 
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve>)*/

#endif
