#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "viennacl.h"
#include "entry_proxy.hpp"

#include <boost/numeric/ublas/vector_sparse.hpp>

#include <viennacl/vector.hpp>
#include <viennacl/vector_proxy.hpp>

namespace ublas = boost::numeric::ublas;

template <class T>
struct vector_to_list_converter
{
  static PyObject* convert(std::vector<T> const& v)
  {
    bp::list l;
    for (std::size_t i = 0; i < v.size(); ++i)
      l.append((T)v[i]);
    
    return bp::incref(l.ptr());
  }
};

// TODO: Obliterate below with the above converter
template <class SCALARTYPE>
bp::list std_vector_to_list(const std::vector<SCALARTYPE>& v)
{
  bp::list l;
  for (unsigned int i = 0; i < v.size(); ++i)
    l.append((SCALARTYPE)v[i]);
  
  return l;
}

template <class SCALARTYPE>
np::ndarray std_vector_to_ndarray(const std::vector<SCALARTYPE>& v)
{
  return np::from_object(std_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_ndarray(const np::ndarray& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }
  
  vcl::vcl_size_t s = (vcl::vcl_size_t) array.shape(0);
  
  std::vector<SCALARTYPE> *v = new std::vector<SCALARTYPE>(s);
  
  for (vcl::vcl_size_t i=0; i < s; ++i)
    (*v)[i] = bp::extract<SCALARTYPE>(array[i]);
  
  return vcl::tools::shared_ptr<std::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_list(const bp::list& l)
{
  return std_vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()));
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<std::vector<SCALARTYPE> >
std_vector_init_scalar(vcl::vcl_size_t length, SCALARTYPE value) {
  std::vector<SCALARTYPE> *v = new std::vector<SCALARTYPE>(length);
  for (vcl::vcl_size_t i=0; i < length; ++i)
    (*v)[i] = value;
  return vcl::tools::shared_ptr<std::vector<SCALARTYPE> >(v);
}

// Vector -- vcl::vector

template <class SCALARTYPE>
bp::list vcl_vector_to_list(const vcl::vector_base<SCALARTYPE>& v)
{
  std::vector<SCALARTYPE> c(v.size());
  vcl::fast_copy(v.begin(), v.end(), c.begin());

  return std_vector_to_list(c);
}

template <class SCALARTYPE>
np::ndarray vcl_vector_to_ndarray(const vcl::vector_base<SCALARTYPE>& v)
{
  return np::from_object(vcl_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_ndarray(const np::ndarray& array)
{
  int d = array.get_nd();
  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }
  
  vcl::vcl_size_t s = (vcl::vcl_size_t) array.shape(0);
  
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(s);
  std::vector<SCALARTYPE> cpu_vector(s);
  
  for (vcl::vcl_size_t i=0; i < s; ++i)
    cpu_vector[i] = bp::extract<SCALARTYPE>(array[i]);
  
  vcl::fast_copy(cpu_vector.begin(), cpu_vector.end(), v->begin());

  return vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_list(const bp::list& l)
{
  return vcl_vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()));
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_scalar(vcl::vcl_size_t length, SCALARTYPE value)
{
  ublas::scalar_vector<SCALARTYPE> s_v(length, value);
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(length);
  vcl::copy(s_v.begin(), s_v.end(), v->begin());
  return vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector_base<SCALARTYPE> >
vcl_range(vcl::vector_base<SCALARTYPE>& vec,
          std::size_t start, std::size_t end)
{
  vcl::range r(start, end);
  vcl::vector_range<vcl::vector_base<SCALARTYPE> > *v_r = new vcl::vector_range
    <vcl::vector_base<SCALARTYPE> >(vec, r);
  return vcl::tools::shared_ptr<vcl::vector_base<SCALARTYPE> >(v_r);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector_base<SCALARTYPE> >
vcl_slice(vcl::vector_base<SCALARTYPE>& vec,
          std::size_t start, std::size_t stride, std::size_t size)
{
  vcl::slice slice(start, stride, size);
  vcl::vector_slice<vcl::vector_base<SCALARTYPE> > *v_s = new vcl::vector_slice
    <vcl::vector_base<SCALARTYPE> >(vec, slice);
  return vcl::tools::shared_ptr<vcl::vector_base<SCALARTYPE> >(v_s);
}

template <class T>
std::vector<T> vector_from_list(bp::list l) {
  std::vector<T> v(bp::len(l));
  for (std::size_t n = 0; n < bp::len(l); n++)
    v.push_back(bp::extract<T>(l[n]));
  return v;
}

template <class T>
bp::list list_from_vector(std::vector<T> v) {
  bp::list l;
  for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++)
    l.append(*it);
  return l;
}

#define EXPORT_VECTOR_CLASS(TYPE)					\
  bp::class_<vcl::vector_base<TYPE>,                                    \
	     vcl::tools::shared_ptr<vcl::vector_base<TYPE> > >               \
    ("vector_base", bp::no_init)                                        \
    .def("get_entry", &get_vcl_vector_entry<TYPE, vcl::vector_base<TYPE> >) \
    .def("set_entry", &set_vcl_vector_entry<TYPE, vcl::vector_base<TYPE> >) \
    .def("as_ndarray", &vcl_vector_to_ndarray<TYPE>)			\
    .def("as_list", &vcl_vector_to_list<TYPE>)                          \
    .add_property("size", &vcl::vector_base<TYPE>::size)                \
    .add_property("internal_size", &vcl::vector_base<TYPE>::internal_size) \
    .add_property("index_norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,	\
		  vcl::vector_base<TYPE>&,                              \
		  op_index_norm_inf, 0>)				\
    ;                                                                   \
  bp::class_<vcl::vector_range<vcl::vector_base<TYPE> >,                \
             vcl::tools::shared_ptr<vcl::vector_range<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ("vector_range", bp::no_init);                                      \
  bp::class_<vcl::vector_slice<vcl::vector_base<TYPE> >,                \
             vcl::tools::shared_ptr<vcl::vector_slice<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ("vector_slice", bp::no_init);                                      \
  bp::class_<vcl::vector<TYPE>,						\
	     vcl::tools::shared_ptr<vcl::vector<TYPE> >,                     \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ( "vector_" #TYPE )                                                 \
    .def(bp::init<int>())						\
    .def(bp::init<vcl::vector_base<TYPE> >())				\
    .def("__init__", bp::make_constructor(vcl_vector_init_ndarray<TYPE>)) \
    .def("__init__", bp::make_constructor(vcl_vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(vcl_vector_init_scalar<TYPE>))\
    ;                                                                   \
  bp::class_<std::vector<TYPE>,						\
	     vcl::tools::shared_ptr<std::vector<TYPE> > >			\
    ( "std_vector_" #TYPE )                                             \
    .def(bp::init<int>())						\
    .def(bp::init<std::vector<TYPE> >())				\
    .def("__init__", bp::make_constructor(std_vector_init_ndarray<TYPE>)) \
    .def("__init__", bp::make_constructor(std_vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(std_vector_init_scalar<TYPE>))\
    .def("as_ndarray", &std_vector_to_ndarray<TYPE>)                    \
    .def("as_list", &std_vector_to_list<TYPE>)                          \
    .add_property("size", &std::vector<TYPE>::size)			\
    ;                                                                   \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_range<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_##TYPE##_range,              \
                            (CONCAT(vcl::vector_base<TYPE>&,            \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_range<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_range_##TYPE##_range,        \
                            (CONCAT(vcl::vector_range<                  \
                                    vcl::vector_base<TYPE> >&,          \
                                    const vcl::range&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_slice<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_##TYPE##_slice,              \
                            (CONCAT(vcl::vector_base<TYPE>&,            \
                                    const vcl::slice&)))                \
  DISAMBIGUATE_FUNCTION_PTR(CONCAT(vcl::vector_slice<                   \
                                   vcl::vector_base<TYPE> >),           \
                            vcl::project,                               \
                            project_vector_slice_##TYPE##_slice,        \
                            (CONCAT(vcl::vector_slice<                  \
                                    vcl::vector_base<TYPE> >&,          \
                                    const vcl::slice&)))                \
  bp::def("project_vector_" #TYPE, project_vector_##TYPE##_range);      \
  bp::def("project_vector_" #TYPE, project_vector_range_##TYPE##_range); \
  bp::def("project_vector_" #TYPE, project_vector_##TYPE##_slice);      \
  bp::def("project_vector_" #TYPE, project_vector_slice_##TYPE##_slice);

#endif
