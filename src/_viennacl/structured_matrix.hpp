#ifndef _PYVIENNACL_STRUCTURED_MATRIX_HPP
#define _PYVIENNACL_STRUCTURED_MATRIX_HPP

#include "dense_matrix.hpp"

#ifdef VIENNACL_WITH_OPENCL

#include <viennacl/circulant_matrix.hpp>
#include <viennacl/hankel_matrix.hpp>
#include <viennacl/toeplitz_matrix.hpp>
#include <viennacl/vandermonde_matrix.hpp>

#endif

template<class MATRIXTYPE>
np::ndarray vcl_structured_matrix_to_ndarray(MATRIXTYPE& m)
{

  // TODO: THIS IS VERY CRUDE!

  typedef typename MATRIXTYPE::value_type::value_type SCALARTYPE;

  ublas::matrix<SCALARTYPE> cpu_dense_m(m.size1(), m.size2());
  vcl::matrix<SCALARTYPE> vcl_dense_m(m.size1(), m.size2());
  vcl::copy(m, cpu_dense_m);
  vcl::copy(cpu_dense_m, vcl_dense_m);
  return vcl_matrix_to_ndarray<SCALARTYPE>(vcl_dense_m);
}

#define EXPORT_STRUCTURED_MATRIX(MAT, SCALAR)                           \
  bp::class_<vcl::MAT<SCALAR, 4>, boost::noncopyable>                   \
  ( #MAT "_" #SCALAR, bp::init<vcl::vcl_size_t, vcl::vcl_size_t>() )    \
  .def("get_entry", &get_vcl_matrix_entry<SCALAR, vcl::MAT<SCALAR, 4> >) \
  .def("set_entry", &set_vcl_matrix_entry<SCALAR, vcl::MAT<SCALAR, 4> >) \
  .def("as_ndarray",                                                    \
       &vcl_structured_matrix_to_ndarray<vcl::MAT<SCALAR, 4> >)         \
  .add_property("size1", &vcl::MAT<SCALAR, 4>::size1)                   \
  .add_property("size2", &vcl::MAT<SCALAR, 4>::size2)                   \
  .add_property("internal_size",                                        \
                  &vcl::MAT<SCALAR, 4>::internal_size)                  \
  ;

#endif
