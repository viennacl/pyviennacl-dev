#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "viennacl.h"

#include <viennacl/scheduler/execute.hpp>

class statement_node_wrapper {

  vcl::scheduler::statement::value_type vcl_node;

public:

  statement_node_wrapper(const statement_node_wrapper& node)
    : vcl_node(node.vcl_node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node node)
    : vcl_node(node)
  { }

  statement_node_wrapper(vcl::scheduler::statement_node_type_family lhs_family,
			 vcl::scheduler::statement_node_subtype lhs_subtype,
			 vcl::scheduler::statement_node_numeric_type lhs_numeric_type,
			 vcl::scheduler::operation_node_type_family op_family,
			 vcl::scheduler::operation_node_type op_type,
			 vcl::scheduler::statement_node_type_family rhs_family,
			 vcl::scheduler::statement_node_subtype rhs_subtype,
			 vcl::scheduler::statement_node_numeric_type rhs_numeric_type)
  {
    vcl_node.op.type_family = op_family;
    vcl_node.op.type = op_type;
    vcl_node.lhs.type_family = lhs_family;
    vcl_node.lhs.subtype = lhs_subtype;
    vcl_node.lhs.numeric_type = lhs_numeric_type;
    vcl_node.rhs.type_family = rhs_family;
    vcl_node.rhs.subtype = rhs_subtype;
    vcl_node.rhs.numeric_type = rhs_numeric_type;
  }

  vcl::scheduler::statement_node& get_vcl_statement_node()
  {
    return vcl_node;
  }

  vcl::scheduler::statement_node get_vcl_statement_node() const
  {
    return vcl_node;
  }

#define SET_OPERAND(T, I)					   \
  void set_operand_to_ ## I (int o, T I) {			   \
    switch (o) {						   \
    case 0:							   \
      vcl_node.lhs.I  = I;					   \
      break;							   \
    case 1:							   \
      vcl_node.rhs.I  = I;					   \
      break;							   \
    default:							   \
      throw vcl::scheduler::statement_not_supported_exception      \
	("Only support operands 0 or 1");			   \
    }								   \
  }

  SET_OPERAND(std::size_t,       node_index)

  SET_OPERAND(char,              host_char)
  SET_OPERAND(unsigned char,     host_uchar)
  SET_OPERAND(short,             host_short)
  SET_OPERAND(unsigned short,    host_ushort)
  SET_OPERAND(int,               host_int)
  SET_OPERAND(unsigned int,      host_uint)
  SET_OPERAND(long,              host_long)
  SET_OPERAND(unsigned long,     host_ulong)
  SET_OPERAND(float,             host_float)
  SET_OPERAND(double,            host_double)

  // NB: need to add remaining scalar types as they become available
  SET_OPERAND(vcl::scalar<float>*, scalar_float)
  SET_OPERAND(vcl::scalar<double>*, scalar_double)

  // NB: need to add remaining vector types as they become available
  SET_OPERAND(vcl::vector_base<float>*, vector_float)
  SET_OPERAND(vcl::vector_base<double>*, vector_double)

  SET_OPERAND(vcl::implicit_vector_base<float>*, implicit_vector_float)
  SET_OPERAND(vcl::implicit_vector_base<double>*, implicit_vector_double)

  // NB: need to add remaining matrix_row types as they become available
  SET_OPERAND(CONCAT(vcl::matrix_base<float, vcl::row_major>*),
              matrix_row_float)
  SET_OPERAND(CONCAT(vcl::matrix_base<double, vcl::row_major>*),
              matrix_row_double)
  
  // NB: need to add remaining matrix_col types as they become available
  SET_OPERAND(CONCAT(vcl::matrix_base<float, vcl::column_major>*),
              matrix_col_float)
  SET_OPERAND(CONCAT(vcl::matrix_base<double, vcl::column_major>*),
              matrix_col_double)

  SET_OPERAND(vcl::implicit_matrix_base<float>*, implicit_matrix_float)
  SET_OPERAND(vcl::implicit_matrix_base<double>*, implicit_matrix_double)

  SET_OPERAND(vcl::compressed_matrix<float>*, compressed_matrix_float)
  SET_OPERAND(vcl::compressed_matrix<double>*, compressed_matrix_double)

  SET_OPERAND(vcl::coordinate_matrix<float>*, coordinate_matrix_float)
  SET_OPERAND(vcl::coordinate_matrix<double>*, coordinate_matrix_double)

  SET_OPERAND(vcl::ell_matrix<float>*, ell_matrix_float)
  SET_OPERAND(vcl::ell_matrix<double>*, ell_matrix_double)

  SET_OPERAND(vcl::hyb_matrix<float>*, hyb_matrix_float)
  SET_OPERAND(vcl::hyb_matrix<double>*, hyb_matrix_double)

};
#undef SET_OPERAND

class statement_wrapper {
  typedef vcl::scheduler::statement::container_type nodes_container_t;

  typedef nodes_container_t::iterator nodes_iterator;
  typedef nodes_container_t::const_iterator nodes_const_iterator;
  
  nodes_container_t vcl_expression_nodes;

public:

  statement_wrapper() {
    vcl_expression_nodes = nodes_container_t(0);
  }

  void execute() {
    vcl::scheduler::statement tmp_statement(vcl_expression_nodes);
    vcl::scheduler::execute(tmp_statement);
  }

  std::size_t size() const {
    return vcl_expression_nodes.size();
  }

  void clear() {
    vcl_expression_nodes.clear();
  }
  
  statement_node_wrapper get_node(std::size_t offset) const {
    return statement_node_wrapper(vcl_expression_nodes[offset]);
  }

  void erase_node(std::size_t offset)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.erase(it+offset);
  }

  void insert_at_index(std::size_t offset,
		       const statement_node_wrapper& node)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.insert(it+offset, node.get_vcl_statement_node());
  }

  void insert_at_begin(const statement_node_wrapper& node)
  {
    nodes_iterator it = vcl_expression_nodes.begin();
    vcl_expression_nodes.insert(it, node.get_vcl_statement_node());
  }

  void insert_at_end(const statement_node_wrapper& node)
  {
    vcl_expression_nodes.push_back(node.get_vcl_statement_node());
  }

};

#endif
