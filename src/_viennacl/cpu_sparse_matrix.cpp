#include "cpu_sparse_matrix.hpp"

PYVCL_SUBMODULE(cpu_sparse_matrix)
{
  EXPORT_CPU_SPARSE_MATRIX(float);
  EXPORT_CPU_SPARSE_MATRIX(double);
  // TODO: other types
}

