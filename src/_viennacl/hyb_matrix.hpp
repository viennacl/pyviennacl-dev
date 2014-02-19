#ifndef _PYVIENNACL_HYB_MATRIX_H
#define _PYVIENNACL_HYB_MATRIX_H

#include "sparse_matrix.h"

#define EXPORT_HYB_MATRIX(TYPE)                                         \
  bp::class_<vcl::hyb_matrix<TYPE>,                                     \
             vcl::tools::shared_ptr<vcl::hyb_matrix<TYPE> >,            \
             boost::noncopyable >                                       \
  ("hyb_matrix", bp::no_init)                                           \
  .add_property("size1",                                                \
                make_function(&vcl::hyb_matrix<TYPE>::size1,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .add_property("size2",                                                \
                make_function(&vcl::hyb_matrix<TYPE>::size2,            \
			      bp::return_value_policy<bp::return_by_value>())) \
  .def("prod", pyvcl_do_2ary_op<vcl::vector<TYPE>,                      \
       vcl::hyb_matrix<TYPE>&, vcl::vector<TYPE>&,                      \
       op_prod, 0>)                                                     \
    ;

#endif
