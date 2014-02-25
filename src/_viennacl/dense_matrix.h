#ifndef _PYVIENNACL_DENSE_MATRIX_H
#define _PYVIENNACL_DENSE_MATRIX_H

#include "viennacl.h"
#include "entry_proxy.hpp"

#include <boost/numeric/ublas/matrix.hpp>

#include <viennacl/matrix.hpp>
#include <viennacl/matrix_proxy.hpp>

namespace ublas = boost::numeric::ublas;

template<class ScalarT>
class ndarray_wrapper
{
  const np::ndarray array; // TODO: Perhaps reference to the wrapped ndarray

public:
  ndarray_wrapper(const np::ndarray& a)
    : array(a)
  { }

  vcl::vcl_size_t size1() const { return array.shape(0); }

  vcl::vcl_size_t size2() const { return array.shape(1); }

  ScalarT operator()(vcl::vcl_size_t row, vcl::vcl_size_t col) const
  {
    return bp::extract<ScalarT>(array[row][col]);
  } 

};

template<class SCALARTYPE, class F>
vcl::tools::shared_ptr<vcl::matrix<SCALARTYPE, F> >
matrix_init_scalar(vcl::vcl_size_t n, vcl::vcl_size_t m, SCALARTYPE value)
{
  ublas::scalar_matrix<SCALARTYPE> s_m(n, m, value);
  ublas::matrix<SCALARTYPE> cpu_m(s_m);
  vcl::matrix<SCALARTYPE, F>* mat = new vcl::matrix<SCALARTYPE, F>(n, m);
  vcl::copy(cpu_m, (*mat));
  return vcl::tools::shared_ptr<vcl::matrix<SCALARTYPE, F> >(mat);
}


/** @brief Creates the matrix from the supplied ndarray */
template<class SCALARTYPE, class F>
vcl::tools::shared_ptr<vcl::matrix<SCALARTYPE, F> >
matrix_init_ndarray(const np::ndarray& array)
{
  int d = array.get_nd();
  if (d != 2) {
    PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
    bp::throw_error_already_set();
  }
  
  ndarray_wrapper<SCALARTYPE> wrapper(array);

  vcl::matrix<SCALARTYPE, F>* mat = new vcl::matrix<SCALARTYPE, F>(wrapper.size1(), wrapper.size2());

  vcl::copy(wrapper, (*mat));
  
  return vcl::tools::shared_ptr<vcl::matrix<SCALARTYPE, F> >(mat);
}

template<class SCALARTYPE>
bp::tuple get_strides(const vcl::matrix_base<SCALARTYPE, vcl::row_major>& m) {
  return bp::make_tuple((m.stride1()*m.internal_size2())*sizeof(SCALARTYPE), m.stride2()*sizeof(SCALARTYPE));
}

template<class SCALARTYPE>
bp::tuple get_strides(const vcl::matrix_base<SCALARTYPE, vcl::column_major>& m) {
  return bp::make_tuple(m.stride1()*sizeof(SCALARTYPE), m.stride2()*m.internal_size1()*sizeof(SCALARTYPE));
}

template<class SCALARTYPE>
std::size_t get_offset(const vcl::matrix_base<SCALARTYPE, vcl::row_major>& m) {
  return m.start1()*m.internal_size2() + m.start2();
}

template<class SCALARTYPE>
std::size_t get_offset(const vcl::matrix_base<SCALARTYPE, 
                       vcl::column_major>& m) {
  return m.start1() + m.start2()*m.internal_size1();
}

template<class SCALARTYPE, class VCL_F>
np::ndarray vcl_matrix_to_ndarray(const vcl::matrix_base<SCALARTYPE, VCL_F>& m)
{

  std::size_t size = m.internal_size1() * m.internal_size2() * sizeof(SCALARTYPE);

  SCALARTYPE* data = (SCALARTYPE*)malloc(size);

  vcl::backend::finish();

  // Read the whole matrix
  vcl::backend::memory_read(m.handle(), 0, size, data);
 
  np::dtype dt = np::dtype::get_builtin<SCALARTYPE>();
  bp::tuple shape = bp::make_tuple(m.size1(), m.size2());

  // Delegate determination of strides and start offset to function templates
  bp::tuple strides = get_strides<SCALARTYPE>(m);
  np::ndarray array = np::from_data(data + get_offset<SCALARTYPE>(m),
                                    dt, shape, strides, bp::object(m));

  //std::cout << "NDARRAY_TEST:\n"
  //        << bp::extract<const char*>(bp::str(array))
  //        << std::endl;

  return array;
}


#define EXPORT_DENSE_MATRIX_CLASS(TYPE, LAYOUT, F, CPU_F)               \
  bp::class_<vcl::matrix_base<TYPE, F>,                                 \
	     vcl::tools::shared_ptr<vcl::matrix_base<TYPE, F> > >            \
    ("matrix_base", bp::no_init)                                        \
    .def("get_entry", &get_vcl_matrix_entry<TYPE, vcl::matrix_base<TYPE, F> >) \
    .def("set_entry", &set_vcl_matrix_entry<TYPE, vcl::matrix_base<TYPE, F> >) \
    .def("as_ndarray", &vcl_matrix_to_ndarray<TYPE, F>)                 \
    .add_property("size1", &vcl::matrix_base<TYPE, F>::size1)           \
    .add_property("internal_size1",                                     \
                  &vcl::matrix_base<TYPE, F>::internal_size1)           \
    .add_property("size2", &vcl::matrix_base<TYPE, F>::size2)           \
    .add_property("internal_size2",                                     \
                  &vcl::matrix_base<TYPE, F>::internal_size2)           \
    .add_property("trans", pyvcl_do_1ary_op<vcl::matrix<TYPE, F>,       \
                  vcl::matrix_base<TYPE, F>&,                           \
                  op_trans, 0>)                                         \
    ;                                                                   \
  bp::class_<vcl::matrix_range<vcl::matrix_base<TYPE, F> >,             \
             vcl::tools::shared_ptr<vcl::matrix_range<vcl::matrix_base<TYPE, \
                                                                  F> > >, \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ("matrix_range", bp::no_init);                                      \
  bp::class_<vcl::matrix_slice<vcl::matrix_base<TYPE, F> >,             \
             vcl::tools::shared_ptr<vcl::matrix_slice<vcl::matrix_base<TYPE, \
                                                                  F> > >, \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ("matrix_slice", bp::no_init);                                      \
  bp::class_<vcl::matrix<TYPE, F>,                                      \
             vcl::tools::shared_ptr<vcl::matrix<TYPE, F> >,                  \
             bp::bases<vcl::matrix_base<TYPE, F> > >                    \
    ( "matrix_" #LAYOUT "_" #TYPE )                                     \
    .def(bp::init<vcl::matrix<TYPE, F> >())                             \
    .def(bp::init<vcl::vcl_size_t, vcl::vcl_size_t>())                                \
    .def("__init__", bp::make_constructor(matrix_init_ndarray<TYPE, F>))\
    .def("__init__", bp::make_constructor(matrix_init_scalar<TYPE, F>)) \
    ;                                                                   \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_range<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_##TYPE##_##LAYOUT##_range_range, \
                            (CONCAT(vcl::matrix_base<TYPE, F>&,         \
                                    const vcl::range&,                  \
                                    const vcl::range&)))                \
 DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_range<                    \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_range_##TYPE##_##LAYOUT##_range_range, \
                            (CONCAT(vcl::matrix_range<                  \
                                    vcl::matrix_base<TYPE, F> >&,       \
                                    const vcl::range&,                  \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_slice<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_##TYPE##_##LAYOUT##_slice_slice, \
                            (CONCAT(vcl::matrix_base<TYPE, F>&,         \
                                    const vcl::slice&,                  \
                                    const vcl::slice&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::matrix_slice<                   \
                                   vcl::matrix_base<TYPE, F> >),        \
                            vcl::project,                               \
                            project_matrix_slice_##TYPE##_##LAYOUT##_slice_slice, \
                            (CONCAT(vcl::matrix_slice<                  \
                                    vcl::matrix_base<TYPE, F> >&,       \
                                    const vcl::slice&,                  \
                                    const vcl::slice&)))                \
  bp::def("project_matrix_" #TYPE, project_matrix_##TYPE##_##LAYOUT##_range_range);   \
  bp::def("project_matrix_" #TYPE, project_matrix_range_##TYPE##_##LAYOUT##_range_range); \
  bp::def("project_matrix_" #TYPE, project_matrix_##TYPE##_##LAYOUT##_slice_slice); \
  bp::def("project_matrix_" #TYPE, project_matrix_slice_##TYPE##_##LAYOUT##_slice_slice);

#endif
