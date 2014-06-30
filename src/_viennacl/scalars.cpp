#include "common.hpp"

template <class HostT>
HostT vcl_scalar_to_host(const vcl::scalar<HostT>& vcl_s)
{
  HostT cpu_s = vcl_s;
  return cpu_s;
}

#define EXPORT_SCALAR_CLASS(TYPE)                                       \
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::scalar<TYPE>,                    \
                                  const vcl::scalar<TYPE>               \
                                  ::handle_type&,                       \
                                  handle,                               \
                                  get_scalar_##TYPE##_handle,           \
                                  () const);                            \
  bp::class_<vcl::scalar<TYPE> >("scalar_" #TYPE)                       \
  .def(bp::init<TYPE>())                                                \
  .def(bp::init<TYPE, vcl::context>())                                  \
  .def(bp::init<vcl::scalar<TYPE> >())                                  \
  .def("to_host", &vcl_scalar_to_host<TYPE>)                            \
  .add_property("handle", bp::make_function                             \
                (get_scalar_##TYPE##_handle,                            \
                 bp::return_internal_reference<>()))                    \
  ;

PYVCL_SUBMODULE(scalars) {

  EXPORT_SCALAR_CLASS(int);
  EXPORT_SCALAR_CLASS(uint);
  EXPORT_SCALAR_CLASS(long);
  EXPORT_SCALAR_CLASS(ulong);
  EXPORT_SCALAR_CLASS(float);
  EXPORT_SCALAR_CLASS(double);

}

