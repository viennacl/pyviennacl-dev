#ifndef _PYVIENNACL_SOLVE_OP_FUNC_H
#define _PYVIENNACL_SOLVE_OP_FUNC_H

#include "viennacl.h"

DO_OP_FUNC(op_solve)
{
  ReturnT r = vcl::linalg::solve(o.operand1, o.operand2,
                                 o.operand3);
  //std::cout << "???????????????????????????? 1: " << o.operand1 << std::endl << std::endl;
  //std::cout << "???????????????????????????? 2: " << o.operand2 << std::endl << std::endl;
  //printf("??? %f\n", r(0));
  return r;
} };

DO_OP_FUNC(op_solve_precond)
{
  return vcl::linalg::solve(o.operand1, o.operand2,
                            o.operand3, o.operand4);
} };

DO_OP_FUNC(op_inplace_solve)
{
  vcl::linalg::inplace_solve(o.operand1, o.operand2,
			     o.operand3);
  return o.operand1;
} };

#endif
