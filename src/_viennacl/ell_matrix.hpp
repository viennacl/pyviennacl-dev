#ifndef _PYVIENNACL_ELL_MATRIX_HPP
#define _PYVIENNACL_ELL_MATRIX_HPP

#include "sparse_matrix.hpp"

// TODO: memory_domain

#define EXPORT_ELL_MATRIX(TYPE)                                         \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ell_matrix<TYPE>,                \
                                  const vcl::ell_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle,                               \
                                  get_ell_matrix_##TYPE##_handle,       \
                                  () const);                            \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ell_matrix<TYPE>,                \
                                  const vcl::ell_matrix<TYPE>           \
                                  ::handle_type&,                       \
                                  handle2,                              \
                                  get_ell_matrix_##TYPE##_handle2,      \
                                  () const);                            \
  bp::class_<vcl::ell_matrix<TYPE>,                                     \
             vcl::tools::shared_ptr<vcl::ell_matrix<TYPE> >,            \
             boost::noncopyable >                                       \
  ("ell_matrix", bp::no_init)                                           \
  /*  .add_property("memory_domain",                                    \
                &vcl::compressed_matrix<TYPE>::memory_context,          \
                &vcl::compressed_matrix<TYPE>::switch_memory_context) */ \
  .add_property("handle", bp::make_function                             \
                (get_ell_matrix_##TYPE##_handle,                        \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle2", bp::make_function                            \
                (get_ell_matrix_##TYPE##_handle2,                       \
                 bp::return_internal_reference<>()))                    \
  .add_property("size1",                                                \
                make_function(&vcl::ell_matrix<TYPE>::size1,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::ell_matrix<TYPE>::size2,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("nnz",                                                  \
                make_function(&vcl::ell_matrix<TYPE>::nnz,              \
			      bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::ell_matrix<TYPE>&, vcl::vector<TYPE>&,                      \
       op_prod>)                                                        \
    ;

#endif

