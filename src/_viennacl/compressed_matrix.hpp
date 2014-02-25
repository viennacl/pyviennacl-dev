#ifndef _PYVIENNACL_COMPRESSED_MATRIX_H
#define _PYVIENNACL_COMPRESSED_MATRIX_H

#include "sparse_matrix.h"

#define EXPORT_COMPRESSED_MATRIX(TYPE)                                  \
  bp::class_<cpu_compressed_matrix_wrapper<TYPE> >                      \
  ("cpu_compressed_matrix_" #TYPE)                                      \
  .def(bp::init<>())                                                    \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t>())                                  \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t, vcl::vcl_size_t>())                        \
  .def(bp::init<cpu_compressed_matrix_wrapper<TYPE> >())                \
  .def(bp::init<vcl::compressed_matrix<TYPE> >())                       \
  /*.def(bp::init<vcl::coordinate_matrix<TYPE> >())                    */ \
  .def(bp::init<vcl::ell_matrix<TYPE> >())                              \
  .def(bp::init<vcl::hyb_matrix<TYPE> >())                              \
  .def(bp::init<np::ndarray>())                                         \
  .add_property("nonzeros", &cpu_compressed_matrix_wrapper<TYPE>::places) \
  .add_property("nnz", &cpu_compressed_matrix_wrapper<TYPE>::nnz)       \
  .add_property("size1", &cpu_compressed_matrix_wrapper<TYPE>::size1)   \
  .add_property("size2", &cpu_compressed_matrix_wrapper<TYPE>::size2)   \
  .def("resize", &cpu_compressed_matrix_wrapper<TYPE>::resize)          \
  .def("set_entry", &cpu_compressed_matrix_wrapper<TYPE>::set_entry)    \
  .def("get_entry", &cpu_compressed_matrix_wrapper<TYPE>::get_entry)    \
  .def("as_ndarray", &cpu_compressed_matrix_wrapper<TYPE>::as_ndarray)  \
  .def("as_compressed_matrix",                                          \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<TYPE> >) \
  .def("as_coordinate_matrix",                                          \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<TYPE> >) \
  .def("as_ell_matrix",                                                 \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix<vcl::ell_matrix<TYPE> >)                  \
  .def("as_hyb_matrix",                                                 \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix<vcl::hyb_matrix<TYPE> >)                  \
  ;                                                                     \
  bp::class_<vcl::compressed_matrix<TYPE>,                              \
             vcl::tools::shared_ptr<vcl::compressed_matrix<TYPE> > >    \
  ("compressed_matrix_" #TYPE, bp::no_init)                             \
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
                                op_prod, 0>)                            \
    ;

      /*    
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::compressed_matrix<TYPE>,
	 vcl::compressed_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)
      */

#endif
