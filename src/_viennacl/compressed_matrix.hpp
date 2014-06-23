#ifndef _PYVIENNACL_COMPRESSED_MATRIX_HPP
#define _PYVIENNACL_COMPRESSED_MATRIX_HPP

#include "sparse_matrix.hpp"

#define EXPORT_COMPRESSED_MATRIX(TYPE)                                  \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::compressed_matrix<TYPE>,         \
                                  vcl::compressed_matrix<TYPE>::handle_type&, \
                                  handle,                               \
                                  get_compressed_matrix_##TYPE##_handle, \
                                  ());                                  \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::compressed_matrix<TYPE>,         \
                                  vcl::compressed_matrix<TYPE>::handle_type&, \
                                  handle1,                              \
                                  get_compressed_matrix_##TYPE##_handle1, \
                                  ());                                  \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::compressed_matrix<TYPE>,         \
                                  vcl::compressed_matrix<TYPE>::handle_type&, \
                                  handle2,                              \
                                  get_compressed_matrix_##TYPE##_handle2, \
                                  ());                                  \
  bp::class_<vcl::compressed_matrix<TYPE>,                              \
             vcl::tools::shared_ptr<vcl::compressed_matrix<TYPE> > >    \
  ("compressed_matrix_" #TYPE, bp::no_init)                             \
  .add_property("memory_domain",                                        \
                &vcl::compressed_matrix<TYPE>::memory_context,          \
                &vcl::compressed_matrix<TYPE>::switch_memory_context)   \
  .add_property("handle", bp::make_function                             \
                (get_compressed_matrix_##TYPE##_handle,                 \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle1", bp::make_function                            \
                (get_compressed_matrix_##TYPE##_handle1,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("handle2", bp::make_function                            \
                (get_compressed_matrix_##TYPE##_handle2,                \
                 bp::return_internal_reference<>()))                    \
  .add_property("size1",                                                \
                make_function(&vcl::compressed_matrix<TYPE>::size1,     \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::compressed_matrix<TYPE>::size2,     \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("nnz",                                                  \
                make_function(&vcl::compressed_matrix<TYPE>::nnz,       \
			      bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,               \
       op_prod>)                                                        \
  ;

      /*    
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve>)
      */

#endif
