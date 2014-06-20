#include "common.hpp"

#define EXPORT_SCALAR_CLASS(TYPE)                                       \
  bp::class_<vcl::scalar<TYPE> >("scalar_" #TYPE)                       \
    .def(bp::init<TYPE>())                                              \
    .def(bp::init<vcl::scalar<TYPE> >())                                \
    .def("to_host", &vcl_scalar_to_host<TYPE>)

PYVCL_SUBMODULE(scalars) {

  EXPORT_SCALAR_CLASS(int);
  EXPORT_SCALAR_CLASS(uint);
  EXPORT_SCALAR_CLASS(long);
  EXPORT_SCALAR_CLASS(ulong);
  EXPORT_SCALAR_CLASS(float);
  EXPORT_SCALAR_CLASS(double);

}

