#ifndef _PYVIENNACL_HYB_MATRIX_HPP
#define _PYVIENNACL_HYB_MATRIX_HPP

#include "sparse_matrix.hpp"

// TODO: memory_domain

#define EXPORT_HYB_MATRIX(TYPE)                                         \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::hyb_matrix<TYPE>,                \
                                  const vcl::hyb_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle,                               \
                                  get_hyb_matrix_##TYPE##_handle,       \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::hyb_matrix<TYPE>,                \
                                  const vcl::hyb_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle2,                              \
                                  get_hyb_matrix_##TYPE##_handle2,      \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::hyb_matrix<TYPE>,                \
                                  const vcl::hyb_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle3,                               \
                                  get_hyb_matrix_##TYPE##_handle3,       \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::hyb_matrix<TYPE>,                \
                                  const vcl::hyb_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle4,                              \
                                  get_hyb_matrix_##TYPE##_handle4,      \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::hyb_matrix<TYPE>,                \
                                  const vcl::hyb_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle5,                              \
                                  get_hyb_matrix_##TYPE##_handle5,      \
                                  () const);                            \
  bp::class_<vcl::hyb_matrix<TYPE>,                                     \
             vcl::tools::shared_ptr<vcl::hyb_matrix<TYPE> >,            \
             boost::noncopyable >                                       \
  ("hyb_matrix", bp::no_init)                                           \
  /*  .add_property("memory_domain",                                    \
                &vcl::compressed_matrix<TYPE>::memory_context,          \
                &vcl::compressed_matrix<TYPE>::switch_memory_context) */ \
  .add_property("handle", bp::make_function                             \
                (get_hyb_matrix_##TYPE##_handle,                        \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle2", bp::make_function                            \
                (get_hyb_matrix_##TYPE##_handle2,                       \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle3", bp::make_function                            \
                (get_hyb_matrix_##TYPE##_handle3,                       \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle4", bp::make_function                            \
                (get_hyb_matrix_##TYPE##_handle4,                       \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle5", bp::make_function                            \
                (get_hyb_matrix_##TYPE##_handle5,                       \
                 bp::return_internal_reference<>()))                    \
  .add_property("size1",                                                \
                make_function(&vcl::hyb_matrix<TYPE>::size1,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::hyb_matrix<TYPE>::size2,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::hyb_matrix<TYPE>&, vcl::vector<TYPE>&,                      \
       op_prod>)                                                        \
    ;

#endif
