#ifndef _PYVIENNACL_SLICED_ELL_MATRIX_HPP
#define _PYVIENNACL_SLICED_ELL_MATRIX_HPP

#include "sparse_matrix.hpp"

// TODO: memory_domain

#define EXPORT_SLICED_ELL_MATRIX(TYPE)                                  \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::sliced_ell_matrix<TYPE>,         \
                                  const vcl::sliced_ell_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle,                               \
                                  get_sliced_ell_matrix_##TYPE##_handle, \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::sliced_ell_matrix<TYPE>,         \
                                  const vcl::sliced_ell_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle1,                              \
                                  get_sliced_ell_matrix_##TYPE##_handle1, \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::sliced_ell_matrix<TYPE>,         \
                                  const vcl::sliced_ell_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle2,                              \
                                  get_sliced_ell_matrix_##TYPE##_handle2, \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::sliced_ell_matrix<TYPE>,         \
                                  const vcl::sliced_ell_matrix<TYPE>    \
                                  ::handle_type&,                       \
                                  handle3,                              \
                                  get_sliced_ell_matrix_##TYPE##_handle3, \
                                  () const);                            \
  bp::class_<vcl::sliced_ell_matrix<TYPE>,                              \
             vcl::tools::shared_ptr<vcl::sliced_ell_matrix<TYPE> >,     \
             boost::noncopyable >                                       \
    ("sliced_ell_matrix", bp::no_init)                                  \
  /*  .add_property("memory_domain",                                    \
                &vcl::compressed_matrix<TYPE>::memory_context,          \
                &vcl::compressed_matrix<TYPE>::switch_memory_context) */ \
  .add_property("handle", bp::make_function                             \
                (get_sliced_ell_matrix_##TYPE##_handle,                 \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle1", bp::make_function                            \
                (get_sliced_ell_matrix_##TYPE##_handle1,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle2", bp::make_function                            \
                (get_sliced_ell_matrix_##TYPE##_handle2,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle3", bp::make_function                            \
                (get_sliced_ell_matrix_##TYPE##_handle3,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("size1",                                                \
                make_function(&vcl::sliced_ell_matrix<TYPE>::size1,     \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::sliced_ell_matrix<TYPE>::size2,     \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("rows_per_block",                                       \
                make_function(&vcl::sliced_ell_matrix<TYPE>::rows_per_block, \
			      bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::sliced_ell_matrix<TYPE>&, vcl::vector<TYPE>&,               \
       op_prod>)                                                        \
    ;

#endif

