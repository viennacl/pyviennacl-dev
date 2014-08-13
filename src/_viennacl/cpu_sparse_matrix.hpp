#ifndef _PYVIENNACL_CPU_SPARSE_MATRIX_HPP
#define _PYVIENNACL_CPU_SPARSE_MATRIX_HPP

#include "sparse_matrix.hpp"
#include <viennacl/tools/matrix_generation.hpp>

template <typename ScalarT>
cpu_sparse_matrix_wrapper<ScalarT>
generate_fdm_laplace(vcl::vcl_size_t points_x, vcl::vcl_size_t points_y) {
  typedef typename cpu_sparse_matrix_wrapper<ScalarT>::ublas_sparse_t ublas_sparse_t;
  ublas_sparse_t A(points_x * points_y, points_x * points_y, points_x * points_y);
  vcl::tools::generate_fdm_laplace<ublas_sparse_t>(A, points_x, points_y);
  return cpu_sparse_matrix_wrapper<ScalarT>(A);
}

#define EXPORT_CPU_SPARSE_MATRIX(TYPE)                                  \
  bp::def("generate_fdm_laplace_" #TYPE, &generate_fdm_laplace<TYPE>);  \
  bp::class_<cpu_sparse_matrix_wrapper<TYPE> >                          \
  ("cpu_sparse_matrix_" #TYPE)                                          \
  .def(bp::init<>())                                                    \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t>())                    \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t, vcl::vcl_size_t>())   \
  .def(bp::init<cpu_sparse_matrix_wrapper<TYPE> >())                    \
  .def(bp::init<vcl::compressed_matrix<TYPE> >())                       \
  /*.def(bp::init<vcl::coordinate_matrix<TYPE> >())                    */ \
  .def(bp::init<vcl::ell_matrix<TYPE> >())                              \
  .def(bp::init<vcl::hyb_matrix<TYPE> >())                              \
  .def(bp::init<np::ndarray>())                                         \
  .add_property("nonzeros", &cpu_sparse_matrix_wrapper<TYPE>::places)   \
  .add_property("nnz", &cpu_sparse_matrix_wrapper<TYPE>::nnz)           \
  .add_property("size1", &cpu_sparse_matrix_wrapper<TYPE>::size1)       \
  .add_property("size2", &cpu_sparse_matrix_wrapper<TYPE>::size2)       \
  .add_property("vcl_context",                                          \
                bp::make_function(&cpu_sparse_matrix_wrapper<TYPE>      \
                                  ::get_vcl_context,                    \
                                  bp::return_value_policy<bp::reference_existing_object>()), \
                &cpu_sparse_matrix_wrapper<TYPE>::set_vcl_context)      \
  .def("resize", &cpu_sparse_matrix_wrapper<TYPE>::resize)              \
  .def("set_entry", &cpu_sparse_matrix_wrapper<TYPE>::set_entry)        \
  .def("get_entry", &cpu_sparse_matrix_wrapper<TYPE>::get_entry)        \
  .def("erase_entry", &cpu_sparse_matrix_wrapper<TYPE>::erase_entry)    \
  .def("insert_entry", &cpu_sparse_matrix_wrapper<TYPE>::insert_entry)  \
  .def("as_ndarray", &cpu_sparse_matrix_wrapper<TYPE>::as_ndarray)      \
  .def("as_compressed_matrix",                                          \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<TYPE> >) \
  .def("as_compressed_compressed_matrix",                               \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix<vcl::compressed_compressed_matrix<TYPE> >) \
  .def("as_coordinate_matrix",                                          \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<TYPE> >) \
  .def("as_ell_matrix",                                                 \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix<vcl::ell_matrix<TYPE> >)                  \
  .def("as_sliced_ell_matrix",                                          \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix<vcl::sliced_ell_matrix<TYPE> >)           \
  .def("as_hyb_matrix",                                                 \
       &cpu_sparse_matrix_wrapper<TYPE>                                 \
       ::as_vcl_sparse_matrix<vcl::hyb_matrix<TYPE> >)                  \
  ;

#endif
