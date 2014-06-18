#ifndef _PYVIENNACL_H
#define _PYVIENNACL_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include <viennacl/tools/shared_ptr.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#define CONCAT(...) __VA_ARGS__

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;

typedef void* NoneT;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

namespace viennacl {
  namespace tools {
    template <typename T> T* get_pointer(vcl::tools::shared_ptr<T> const& p) {
      return p.get();
    }
  }
}

namespace boost {
  namespace python {
    template <typename T> struct pointee<vcl::tools::shared_ptr<T> > {
      typedef T type;
    };
  }
}

// TODO: Use ViennaCL operation tags?
enum op_t {
  op_inner_prod=0,   //  0
  op_outer_prod,     //  1
  op_element_pow,    //  2
  op_norm_1,         //  3
  op_norm_2,         //  4
  op_norm_inf,       //  5
  op_norm_frobenius, //  6
  op_index_norm_inf, //  7
  op_plane_rotation, //  8
  op_prod,           //  9
  op_solve,          // 10
  op_solve_precond,  // 11
  op_inplace_solve,  // 12
  op_fft,            // 13
  op_ifft,           // 14
  op_inplace_fft,    // 15
  op_inplace_ifft,   // 16
  op_inplace_qr,     // 17
  op_inplace_qr_apply_trans_q, // 18
  op_recoverq,       // 19
  op_nmf             // 20
};

// Generic operation dispatch class -- see specialisations below
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
struct pyvcl_worker
{
  static ReturnT do_op(void* o) {}
};

// This class wraps operations in a type-independent way up to 4 operands.
// It's mainly used to simplify and consolidate calling conventions in the 
// main module code far below, but it also includes a small amount of logic
// for the extraction of C++ types from Python objects where necessary.
//
// Ultimately, I may well do away with this, and interface with the kernel
// scheduler directly. But this is a useful start to have, in order to get
// a working prototype.
//
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs=0>
struct pyvcl_op
{
  Operand1T operand1;
  Operand2T operand2;
  Operand3T operand3;
  Operand4T operand4;
  friend struct pyvcl_worker<ReturnT,
			     Operand1T, Operand2T,
			     Operand3T, Operand4T,
			     op, PyObjs>;
  
  pyvcl_op(Operand1T opand1, Operand2T opand2,
	   Operand3T opand3, Operand4T opand4)
    : operand1(opand1), operand2(opand2),
      operand3(opand3), operand4(opand4)
  {
    
    /*
      
      The value of the template variable PyObjs determines which operands
      need to be extracted from Python objects, by coding the operand
      "position" in binary. This is the object-extraction logic alluded to
      in the comments above.
      
      So, given (as an example) PyObjs == 7 == 0111b, and given that we 
      number operands from left to right, the following operands need
      extraction: operand2, operand3, and operand4.
      
    */

    /*    
    if (PyObjs & 8) {
      operand1 = static_cast<Operand1T>
	(bp::extract<Operand1T>((bp::api::object)opand1));
    }
    
    if (PyObjs & 4) {
      operand2 = static_cast<Operand2T>
	(bp::extract<Operand2T>((bp::api::object)opand2));
    }
    
    if (PyObjs & 2) {
      operand3 = static_cast<Operand3T>
	(bp::extract<Operand3T>((bp::api::object)opand3));
    }
    
    if (PyObjs & 1) {
      operand4 = static_cast<Operand4T>
	(bp::extract<Operand4T>((bp::api::object)opand4));
    }
    //*/
  }    

  // Should I just use operator(), I wonder..
  ReturnT do_op()
  {
    return pyvcl_worker<ReturnT,
			Operand1T, Operand2T,
			Operand3T, Operand4T,
			op, PyObjs>::do_op(*this);
  }
};

// Convenient operation dispatch functions.
// These functions make setting up and calling the pyvcl_op class much
// simpler for the specific 1-, 2-, 3- and 4-operand cases.

template <class ReturnT,
	  class Operand1T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_1ary_op(Operand1T a)
{
  pyvcl_op<ReturnT,
	   Operand1T, NoneT,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, NULL, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_2ary_op(Operand1T a, Operand2T b)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, b, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_3ary_op(Operand1T a, Operand2T b, Operand3T c)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   Operand3T, NoneT,
	   op, PyObjs>
    o (a, b, c, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_4ary_op(Operand1T a, Operand2T b,
			 Operand3T c, Operand4T d)
  {
    pyvcl_op<ReturnT,
	     Operand1T, Operand2T,
	     Operand3T, Operand4T,
	     op, PyObjs>
      o (a, b, c, d);
  return o.do_op();
}

// These macros define specialisations of the pyvcl_worker class
// which is used to dispatch viennacl operations.

#define OP_TEMPLATE template <class ReturnT,                    \
                              class Operand1T, class Operand2T, \
                              class Operand3T, class Operand4T, \
                              int PyObjs>
#define PYVCL_WORKER_STRUCT(OP) OP_TEMPLATE                \
  struct pyvcl_worker<ReturnT,                             \
                      Operand1T, Operand2T,                \
                      Operand3T, Operand4T,                \
                      OP, PyObjs>
#define DO_OP_FUNC(OP) PYVCL_WORKER_STRUCT(OP) {                     \
  static ReturnT do_op(pyvcl_op<ReturnT,                             \
                       Operand1T, Operand2T,                         \
                       Operand3T, Operand4T,                         \
                       OP, PyObjs>& o)

DO_OP_FUNC(op_prod)
{
  return vcl::linalg::prod(o.operand1, o.operand2);
} };

/** @brief Returns a double describing the VCL_T */
template <class HostT>
HostT vcl_scalar_to_host(const vcl::scalar<HostT>& vcl_s)
{
  HostT cpu_s = vcl_s;
  return cpu_s;
}

#define DISAMBIGUATE_FUNCTION_PTR(RET, OLD_NAME, NEW_NAME, ARGS) \
  RET (*NEW_NAME) ARGS = &OLD_NAME;

#define DISAMBIGUATE_CLASS_FUNCTION_PTR(CLASS, RET, OLD_NAME, NEW_NAME, ARGS)\
  RET (CLASS::*NEW_NAME) ARGS = &CLASS::OLD_NAME;


/**************
   Submodules
 **************/

#define PYVCL_SUBMODULE(NAME) void export_ ## NAME ()
#define PYTHON_SCOPE_SUBMODULE(NAME)                                    \
  bp::object NAME##_submodule                                           \
  (bp::handle<>(bp::borrowed(PyImport_AddModule("_viennacl." #NAME)))); \
  bp::scope().attr(#NAME) = NAME##_submodule;                           \
  bp::scope NAME##_scope = NAME##_submodule;

PYVCL_SUBMODULE(vector_int);
PYVCL_SUBMODULE(vector_long);
PYVCL_SUBMODULE(vector_uint);
PYVCL_SUBMODULE(vector_ulong);
PYVCL_SUBMODULE(vector_float);
PYVCL_SUBMODULE(vector_double);

PYVCL_SUBMODULE(dense_matrix_int);
PYVCL_SUBMODULE(dense_matrix_long);
PYVCL_SUBMODULE(dense_matrix_uint);
PYVCL_SUBMODULE(dense_matrix_ulong);
PYVCL_SUBMODULE(dense_matrix_float);
PYVCL_SUBMODULE(dense_matrix_double);

PYVCL_SUBMODULE(structured_matrices);

PYVCL_SUBMODULE(compressed_matrix);
PYVCL_SUBMODULE(coordinate_matrix);
PYVCL_SUBMODULE(ell_matrix);
PYVCL_SUBMODULE(hyb_matrix);

PYVCL_SUBMODULE(preconditioners);
PYVCL_SUBMODULE(direct_solvers);
PYVCL_SUBMODULE(iterative_solvers);

PYVCL_SUBMODULE(extra_functions);
PYVCL_SUBMODULE(eig);
PYVCL_SUBMODULE(bandwidth_reduction);

PYVCL_SUBMODULE(scheduler);
PYVCL_SUBMODULE(opencl_support);

#endif
