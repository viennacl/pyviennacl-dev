#ifndef _PYVIENNACL_ELL_MATRIX_H
#define _PYVIENNACL_ELL_MATRIX_H

#include "sparse_matrix.h"

#define EXPORT_ELL_MATRIX(TYPE)                                         \
  bp::class_<vcl::ell_matrix<TYPE>,                                     \
             vcl::tools::shared_ptr<vcl::ell_matrix<TYPE> >,            \
             boost::noncopyable >                                       \
  ("ell_matrix", bp::no_init)                                           \
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
       op_prod, 0>)                                                     \
    ;

#endif

