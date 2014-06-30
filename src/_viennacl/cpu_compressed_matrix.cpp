#include "cpu_compressed_matrix.hpp"

PYVCL_SUBMODULE(cpu_compressed_matrix)
{
  EXPORT_CPU_COMPRESSED_MATRIX(float);
  EXPORT_CPU_COMPRESSED_MATRIX(double);
  // TODO: other types
}

