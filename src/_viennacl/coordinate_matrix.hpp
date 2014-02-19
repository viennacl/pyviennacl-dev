#ifndef _PYVIENNACL_COORDINATE_MATRIX_H
#define _PYVIENNACL_COORDINATE_MATRIX_H

#include "sparse_matrix.h"

#define EXPORT_COORDINATE_MATRIX(TYPE)                                  \
  bp::class_<vcl::coordinate_matrix<TYPE>,                              \
             vcl::tools::shared_ptr<vcl::coordinate_matrix<TYPE> >,     \
             boost::noncopyable >                                       \
  ("coordinate_matrix_" #TYPE, bp::no_init)                             \
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
       op_prod, 0>)                                                     \
  ;
    /* 
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_lower_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::unit_upper_tag,
	 op_inplace_solve, 0>)
    .def("inplace_solve", pyvcl_do_3ary_op<vcl::coordinate_matrix<TYPE>,
	 vcl::coordinate_matrix<TYPE>&, vcl::vector<TYPE>&,
	 vcl::linalg::upper_tag,
	 op_inplace_solve, 0>)*/

#endif
