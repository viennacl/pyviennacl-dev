#include "scheduler.hpp"

PYVCL_SUBMODULE(scheduler)
{

  bp::enum_<vcl::scheduler::operation_node_type_family>
    ("operation_node_type_family")
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_TYPE_FAMILY)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_TYPE_FAMILY)
    ;

  bp::enum_<vcl::scheduler::operation_node_type>("operation_node_type")
    // unary expression
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_ABS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_ACOS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_ASIN_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_ATAN_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_CEIL_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_COS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_COSH_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_EXP_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_FABS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_FLOOR_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_LOG_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_LOG10_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_SIN_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_SINH_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_SQRT_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_TAN_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_TANH_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_TRANS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_NORM_1_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_NORM_2_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_UNARY_NORM_INF_TYPE)
    
    // binary expression
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_ACCESS_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_ASSIGN_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_ADD_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_INPLACE_SUB_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_ADD_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_SUB_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_MAT_VEC_PROD_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_MAT_MAT_PROD_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_MULT_TYPE)// scalar * vector/matrix
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_DIV_TYPE) // vector/matrix / scalar
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_PROD_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_ELEMENT_DIV_TYPE)
    ENUM_VALUE(vcl::scheduler, OPERATION_BINARY_INNER_PROD_TYPE)
    ;

  bp::enum_<vcl::scheduler::statement_node_type_family>
    ("statement_node_type_family")
    ENUM_VALUE(vcl::scheduler, INVALID_TYPE_FAMILY)
    ENUM_VALUE(vcl::scheduler, COMPOSITE_OPERATION_FAMILY)
    ENUM_VALUE(vcl::scheduler, SCALAR_TYPE_FAMILY)
    ENUM_VALUE(vcl::scheduler, VECTOR_TYPE_FAMILY)
    ENUM_VALUE(vcl::scheduler, MATRIX_TYPE_FAMILY)
    ;

  bp::enum_<vcl::scheduler::statement_node_subtype>
    ("statement_node_subtype")
    ENUM_VALUE(vcl::scheduler, INVALID_SUBTYPE)

    ENUM_VALUE(vcl::scheduler, HOST_SCALAR_TYPE)
    ENUM_VALUE(vcl::scheduler, DEVICE_SCALAR_TYPE)

    ENUM_VALUE(vcl::scheduler, DENSE_VECTOR_TYPE)
    ENUM_VALUE(vcl::scheduler, IMPLICIT_VECTOR_TYPE)

    ENUM_VALUE(vcl::scheduler, DENSE_MATRIX_TYPE)
    ENUM_VALUE(vcl::scheduler, IMPLICIT_MATRIX_TYPE)

    ENUM_VALUE(vcl::scheduler, COMPRESSED_MATRIX_TYPE)
    ENUM_VALUE(vcl::scheduler, COORDINATE_MATRIX_TYPE)
    ENUM_VALUE(vcl::scheduler, ELL_MATRIX_TYPE)
    ENUM_VALUE(vcl::scheduler, HYB_MATRIX_TYPE)
    ;

  bp::enum_<vcl::scheduler::statement_node_numeric_type>
    ("statement_node_numeric_type")
    ENUM_VALUE(vcl::scheduler, INVALID_NUMERIC_TYPE)

    ENUM_VALUE(vcl::scheduler, CHAR_TYPE)
    ENUM_VALUE(vcl::scheduler, UCHAR_TYPE)
    ENUM_VALUE(vcl::scheduler, SHORT_TYPE)
    ENUM_VALUE(vcl::scheduler, USHORT_TYPE)
    ENUM_VALUE(vcl::scheduler, INT_TYPE)
    ENUM_VALUE(vcl::scheduler, UINT_TYPE)
    ENUM_VALUE(vcl::scheduler, LONG_TYPE)
    ENUM_VALUE(vcl::scheduler, ULONG_TYPE)
    ENUM_VALUE(vcl::scheduler, HALF_TYPE)
    ENUM_VALUE(vcl::scheduler, FLOAT_TYPE)
    ENUM_VALUE(vcl::scheduler, DOUBLE_TYPE)
    ;

  bp::class_<vcl::scheduler::lhs_rhs_element>("lhs_rhs_element")
    .def_readonly("type_family", &vcl::scheduler::lhs_rhs_element::type_family)
    .def_readonly("subtype", &vcl::scheduler::lhs_rhs_element::subtype)
    .def_readonly("numeric_type", &vcl::scheduler::lhs_rhs_element::numeric_type)
    ;

  bp::class_<vcl::scheduler::op_element>("op_element")
    .def_readonly("type_family", &vcl::scheduler::op_element::type_family)
    .def_readonly("type", &vcl::scheduler::op_element::type)
    ;

  bp::class_<vcl::scheduler::statement_node>("vcl_statement_node")
    .def_readonly("lhs", &vcl::scheduler::statement_node::lhs)
    .def_readonly("rhs", &vcl::scheduler::statement_node::rhs)
    .def_readonly("op", &vcl::scheduler::statement_node::op)
    ;


#define STRINGIFY(S) #S
#define SET_OPERAND(I)					\
  .def(STRINGIFY(set_operand_to_ ## I),			\
       &statement_node_wrapper::set_operand_to_ ## I)

DISAMBIGUATE_CLASS_FUNCTION_PTR(statement_node_wrapper,         // class
                                vcl::scheduler::statement_node, // ret. type
                                get_vcl_statement_node,         // old_name
                                get_vcl_statement_node,         // new_name
                                () const)                       // args

  bp::class_<statement_node_wrapper>("statement_node",
				     bp::init<statement_node_wrapper>())
    .def(bp::init<vcl::scheduler::statement_node_type_family,  // lhs
	 vcl::scheduler::statement_node_subtype,               // lhs
	 vcl::scheduler::statement_node_numeric_type,          // lhs
	 vcl::scheduler::operation_node_type_family,           // op
	 vcl::scheduler::operation_node_type,                  // op
	 vcl::scheduler::statement_node_type_family,           // rhs
	 vcl::scheduler::statement_node_subtype,               // rhs
	 vcl::scheduler::statement_node_numeric_type>())       // rhs
    SET_OPERAND(node_index)
    SET_OPERAND(host_char)
    SET_OPERAND(host_uchar)
    SET_OPERAND(host_short)
    SET_OPERAND(host_ushort)
    SET_OPERAND(host_int)
    SET_OPERAND(host_uint)
    SET_OPERAND(host_long)
    SET_OPERAND(host_ulong)
    SET_OPERAND(host_float)
    SET_OPERAND(host_double)

    SET_OPERAND(scalar_float)
    SET_OPERAND(scalar_double)
    SET_OPERAND(vector_float)
    SET_OPERAND(vector_double)
    SET_OPERAND(implicit_vector_float)
    SET_OPERAND(implicit_vector_double)
    SET_OPERAND(matrix_float)
    SET_OPERAND(matrix_double)
    SET_OPERAND(implicit_matrix_float)
    SET_OPERAND(implicit_matrix_double)
    SET_OPERAND(implicit_vector_float)
    SET_OPERAND(implicit_vector_double)
    SET_OPERAND(compressed_matrix_float)
    SET_OPERAND(compressed_matrix_double)
    SET_OPERAND(coordinate_matrix_float)
    SET_OPERAND(coordinate_matrix_double)
    SET_OPERAND(ell_matrix_float)
    SET_OPERAND(ell_matrix_double)
    SET_OPERAND(hyb_matrix_float)
    SET_OPERAND(hyb_matrix_double)

    .add_property("vcl_statement_node",
	 bp::make_function(get_vcl_statement_node,
			   bp::return_value_policy<bp::return_by_value>()))
    ;

#undef SET_OPERAND

  bp::class_<statement_wrapper>("statement")
    .add_property("size", &statement_wrapper::size)
    .def("execute", &statement_wrapper::execute)
    .def("clear", &statement_wrapper::clear)
    .def("erase_node", &statement_wrapper::erase_node)
    .def("get_node", &statement_wrapper::get_node)
    .def("insert_at_index", &statement_wrapper::insert_at_index)
    .def("insert_at_begin", &statement_wrapper::insert_at_begin)
    .def("insert_at_end", &statement_wrapper::insert_at_end)
    ;

}
