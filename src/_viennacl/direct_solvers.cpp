#include "direct_solvers.hpp"

PYVCL_SUBMODULE(direct_solvers)
{
  EXPORT_DIRECT_SOLVERS(float);
  EXPORT_DIRECT_SOLVERS(double);
  EXPORT_DIRECT_SOLVERS(int);
  EXPORT_DIRECT_SOLVERS(uint);
  EXPORT_DIRECT_SOLVERS(long);
  EXPORT_DIRECT_SOLVERS(ulong);
}

