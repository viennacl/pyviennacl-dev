#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "entry_proxy.hpp"
#include "dense_matrix.hpp"

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
  vcl::copy(v.begin(), v.end(), c.begin());

  return std_vector_to_list(c);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<std::vector<SCALARTYPE> >
vcl_vector_to_std_vector(const vcl::vector_base<SCALARTYPE>& v)
{
  std::vector<SCALARTYPE> *c = new std::vector<SCALARTYPE>(v.size());
  vcl::copy(v.begin(), v.end(), c->begin());
  return vcl::tools::shared_ptr<std::vector<SCALARTYPE> >(c);
}

template <class SCALARTYPE>
np::ndarray vcl_vector_to_ndarray(const vcl::vector_base<SCALARTYPE>& v)
{
  return np::from_object(vcl_vector_to_list<SCALARTYPE>(v),
			 np::dtype::get_builtin<SCALARTYPE>());
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_mem(SCALARTYPE* ptr, vcl::memory_types mem_type,
                    vcl::vcl_size_t size, vcl::vcl_size_t start = 0,
                    vcl::vcl_ptrdiff_t stride = 1)
{
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>
    (ptr, mem_type, size, start, stride);
  return vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_ndarray(const np::ndarray& array, const vcl::context& ctx)
{

  int d = array.get_nd();

  if (d != 1) {
    PyErr_SetString(PyExc_TypeError, "Can only create a vector from a 1-D array!");
    bp::throw_error_already_set();
  }

  vcl::vcl_size_t s = (vcl::vcl_size_t) array.shape(0);

  std::vector<SCALARTYPE> cpu_vector(s);
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(s, ctx);

  for (vcl::vcl_size_t i=0; i < s; ++i)
    cpu_vector[i] = bp::extract<SCALARTYPE>(array[i]);

  vcl::copy(cpu_vector.begin(), cpu_vector.end(), v->begin());

  return vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_list(const bp::list& l, const vcl::context& ctx)
{
  return vcl_vector_init_ndarray<SCALARTYPE>
    (np::from_object(l, np::dtype::get_builtin<SCALARTYPE>()), ctx);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_scalar(vcl::vcl_size_t length, SCALARTYPE value,
                       const vcl::context &ctx)
{
  ublas::scalar_vector<SCALARTYPE> s_v(length, value);
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(length, ctx);
  vcl::copy(s_v.begin(), s_v.end(), v->begin());
  return vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >(v);
}

template <class SCALARTYPE>
vcl::tools::shared_ptr<vcl::vector<SCALARTYPE> >
vcl_vector_init_std_vector(const std::vector<SCALARTYPE>& cpu_vec,
                           const vcl::context &ctx)
{
  vcl::vector<SCALARTYPE> *v = new vcl::vector<SCALARTYPE>(cpu_vec.size(), ctx);
  vcl::copy(cpu_vec.begin(), cpu_vec.end(), v->begin());
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

DO_OP_FUNC(op_index_norm_inf)
{
  return vcl::linalg::index_norm_inf(o.operand1);
}
CLOSE_OP_FUNC;

#define EXPORT_VECTOR_CLASS(TYPE)					\
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::vector_base<TYPE>,               \
                                  vcl::vector_base<TYPE>::handle_type&, \
                                  handle, get_vector_##TYPE##_handle, ()); \
  bp::class_<vcl::vector_base<TYPE>,                                    \
             vcl::tools::shared_ptr<vcl::vector_base<TYPE> > >          \
  ("vector_base",                                                       \
   bp::init<vcl::backend::mem_handle&,                                  \
   vcl::vector_base<TYPE>::size_type,                                   \
   vcl::vector_base<TYPE>::size_type,                                   \
   vcl::vector_base<TYPE>::difference_type>())                          \
    .def("get_entry", &get_vcl_vector_entry<TYPE, vcl::vector_base<TYPE> >) \
    .def("set_entry", &set_vcl_vector_entry<TYPE, vcl::vector_base<TYPE> >) \
    .def("as_ndarray", &vcl_vector_to_ndarray<TYPE>)			\
    .def("as_list", &vcl_vector_to_list<TYPE>)                          \
    .def("as_std_vector", &vcl_vector_to_std_vector<TYPE>)              \
  /*.def("switch_memory_context", &vcl::vector_base<TYPE>::switch_memory_context) */  \
    .add_property("memory_domain", &vcl::vector_base<TYPE>::memory_domain) \
    .add_property("handle", bp::make_function                           \
                  (get_vector_##TYPE##_handle,                          \
                   bp::return_internal_reference<>()))                  \
    .add_property("size", &vcl::vector_base<TYPE>::size)                \
    .add_property("internal_size", &vcl::vector_base<TYPE>::internal_size) \
    .add_property("start", &vcl::vector_base<TYPE>::start)              \
    .add_property("stride", &vcl::vector_base<TYPE>::stride)            \
    .add_property("index_norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,	\
		  vcl::vector_base<TYPE>&,                              \
		  op_index_norm_inf>)                                   \
    ;                                                                   \
  bp::class_<vcl::vector_range<vcl::vector_base<TYPE> >,                \
             vcl::tools::shared_ptr<vcl::vector_range<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
  ("vector_range_" #TYPE, bp::no_init);                                 \
  bp::class_<vcl::vector_slice<vcl::vector_base<TYPE> >,                \
             vcl::tools::shared_ptr<vcl::vector_slice<vcl::vector_base<TYPE> > >, \
             bp::bases<vcl::vector_base<TYPE> > >                       \
  ("vector_slice_" #TYPE, bp::no_init);                                 \
  bp::class_<vcl::vector<TYPE>,						\
	     vcl::tools::shared_ptr<vcl::vector<TYPE> >,                \
             bp::bases<vcl::vector_base<TYPE> > >                       \
    ( "vector_" #TYPE )                                                 \
    .def(bp::init<int>())						\
    .def(bp::init<int, vcl::context>())                                 \
    .def(bp::init<vcl::vector_base<TYPE> >())				\
    .def("__init__", bp::make_constructor(vcl_vector_init_mem<TYPE>))   \
    .def("__init__", bp::make_constructor(vcl_vector_init_ndarray<TYPE>)) \
    .def("__init__", bp::make_constructor(vcl_vector_init_list<TYPE>))	\
    .def("__init__", bp::make_constructor(vcl_vector_init_scalar<TYPE>))\
    .def("__init__", bp::make_constructor(vcl_vector_init_std_vector<TYPE>)) \
    ;                                                                   \
  bp::class_<std::vector<TYPE>,						\
	     vcl::tools::shared_ptr<std::vector<TYPE> > >			\
    ( "std_vector_" #TYPE )                                             \
    .def(bp::init<int>())						\
  .def(bp::init<std::vector<TYPE> >())                                  \
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
