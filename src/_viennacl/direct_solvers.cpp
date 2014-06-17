#include "direct_solvers.hpp"

PYVCL_SUBMODULE(direct_solvers)
{
  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  EXPORT_DIRECT_SOLVERS(float);
  EXPORT_DIRECT_SOLVERS(double);
  EXPORT_DIRECT_SOLVERS(int);
  EXPORT_DIRECT_SOLVERS(uint);
  EXPORT_DIRECT_SOLVERS(long);
  EXPORT_DIRECT_SOLVERS(ulong);
}

