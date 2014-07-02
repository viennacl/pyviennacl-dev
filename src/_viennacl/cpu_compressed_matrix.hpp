#ifndef _PYVIENNACL_CPU_COMPRESSED_MATRIX_HPP
#define _PYVIENNACL_CPU_COMPRESSED_MATRIX_HPP

#include "sparse_matrix.hpp"

#define EXPORT_CPU_COMPRESSED_MATRIX(TYPE)                              \
  bp::class_<cpu_compressed_matrix_wrapper<TYPE> >                      \
  ("cpu_compressed_matrix_" #TYPE)                                      \
  .def(bp::init<>())                                                    \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t>())                    \
  .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t, vcl::vcl_size_t>())   \
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
  .add_property("vcl_context",                                          \
                bp::make_function(&cpu_compressed_matrix_wrapper<TYPE>  \
                                  ::get_vcl_context,                    \
                                  bp::return_internal_reference<>()),   \
                &cpu_compressed_matrix_wrapper<TYPE>::set_vcl_context)  \
  .def("resize", &cpu_compressed_matrix_wrapper<TYPE>::resize)          \
  .def("set_entry", &cpu_compressed_matrix_wrapper<TYPE>::set_entry)    \
  .def("get_entry", &cpu_compressed_matrix_wrapper<TYPE>::get_entry)    \
  .def("erase_entry", &cpu_compressed_matrix_wrapper<TYPE>::erase_entry) \
  .def("insert_entry", &cpu_compressed_matrix_wrapper<TYPE>::insert_entry) \
  .def("as_ndarray", &cpu_compressed_matrix_wrapper<TYPE>::as_ndarray)  \
  .def("as_compressed_matrix",                                          \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<TYPE> >) \
  .def("as_compressed_compressed_matrix",                               \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix<vcl::compressed_compressed_matrix<TYPE> >) \
  .def("as_coordinate_matrix",                                          \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<TYPE> >) \
  .def("as_ell_matrix",                                                 \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix<vcl::ell_matrix<TYPE> >)                  \
  .def("as_hyb_matrix",                                                 \
       &cpu_compressed_matrix_wrapper<TYPE>                             \
       ::as_vcl_sparse_matrix<vcl::hyb_matrix<TYPE> >)                  \
  ;

#endif
