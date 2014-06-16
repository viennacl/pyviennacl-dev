#include "iterative_solvers.hpp"
#include "preconditioners.hpp"

PYVCL_SUBMODULE(iterative_solvers)
{
  /*
  EXPORT_DENSE_ITERATIVE_SOLVERS(float);
  EXPORT_DENSE_ITERATIVE_SOLVERS(double);
  /...* These don't really make sense:
  EXPORT_DENSE_ITERATIVE_SOLVERS(int);
  EXPORT_DENSE_ITERATIVE_SOLVERS(uint);
  EXPORT_DENSE_ITERATIVE_SOLVERS(long);
  EXPORT_DENSE_ITERATIVE_SOLVERS(ulong);
  */

  EXPORT_SPARSE_ITERATIVE_SOLVERS(float);
  EXPORT_SPARSE_ITERATIVE_SOLVERS(double);

}

