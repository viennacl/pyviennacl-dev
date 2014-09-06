#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "common.hpp"
#include <viennacl/scheduler/execute.hpp>
#include <viennacl/scheduler/io.hpp>

#ifdef VIENNACL_WITH_OPENCL
#include <atidlas/execute.hpp>
#endif

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

  void print_vcl_statement_node() const {
    std::cout << get_vcl_statement_node() << std::endl;
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

  SET_OPERAND(vcl::scalar<char>*,           scalar_char)
  SET_OPERAND(vcl::scalar<unsigned char>*,  scalar_uchar)
  SET_OPERAND(vcl::scalar<short>*,          scalar_short)
  SET_OPERAND(vcl::scalar<unsigned short>*, scalar_ushort)
  SET_OPERAND(vcl::scalar<int>*,            scalar_int)
  SET_OPERAND(vcl::scalar<unsigned int>*,   scalar_uint)
  SET_OPERAND(vcl::scalar<long>*,           scalar_long)
  SET_OPERAND(vcl::scalar<unsigned long>*,  scalar_ulong)
  SET_OPERAND(vcl::scalar<float>*,          scalar_float)
  SET_OPERAND(vcl::scalar<double>*,         scalar_double)

  SET_OPERAND(vcl::vector_base<char>*,           vector_char)
  SET_OPERAND(vcl::vector_base<unsigned char>*,  vector_uchar)
  SET_OPERAND(vcl::vector_base<short>*,          vector_short)
  SET_OPERAND(vcl::vector_base<unsigned short>*, vector_ushort)
  SET_OPERAND(vcl::vector_base<int>*,            vector_int)
  SET_OPERAND(vcl::vector_base<unsigned int>*,   vector_uint)
  SET_OPERAND(vcl::vector_base<long>*,           vector_long)
  SET_OPERAND(vcl::vector_base<unsigned long>*,  vector_ulong)
  SET_OPERAND(vcl::vector_base<float>*,          vector_float)
  SET_OPERAND(vcl::vector_base<double>*,         vector_double)

  SET_OPERAND(vcl::implicit_vector_base<char>*,           implicit_vector_char)
  SET_OPERAND(vcl::implicit_vector_base<unsigned char>*,  implicit_vector_uchar)
  SET_OPERAND(vcl::implicit_vector_base<short>*,          implicit_vector_short)
  SET_OPERAND(vcl::implicit_vector_base<unsigned short>*, implicit_vector_ushort)
  SET_OPERAND(vcl::implicit_vector_base<int>*,            implicit_vector_int)
  SET_OPERAND(vcl::implicit_vector_base<unsigned int>*,   implicit_vector_uint)
  SET_OPERAND(vcl::implicit_vector_base<long>*,           implicit_vector_long)
  SET_OPERAND(vcl::implicit_vector_base<unsigned long>*,  implicit_vector_ulong)
  SET_OPERAND(vcl::implicit_vector_base<float>*,          implicit_vector_float)
  SET_OPERAND(vcl::implicit_vector_base<double>*,         implicit_vector_double)

  SET_OPERAND(vcl::matrix_base<char>*,           matrix_char)
  SET_OPERAND(vcl::matrix_base<unsigned char>*,  matrix_uchar)
  SET_OPERAND(vcl::matrix_base<short>*,          matrix_short)
  SET_OPERAND(vcl::matrix_base<unsigned short>*, matrix_ushort)
  SET_OPERAND(vcl::matrix_base<int>*,            matrix_int)
  SET_OPERAND(vcl::matrix_base<unsigned int>*,   matrix_uint)
  SET_OPERAND(vcl::matrix_base<long>*,           matrix_long)
  SET_OPERAND(vcl::matrix_base<unsigned long>*,  matrix_ulong)
  SET_OPERAND(vcl::matrix_base<float>*,          matrix_float)
  SET_OPERAND(vcl::matrix_base<double>*,         matrix_double)

  SET_OPERAND(vcl::implicit_matrix_base<char>*,           implicit_matrix_char)
  SET_OPERAND(vcl::implicit_matrix_base<unsigned char>*,  implicit_matrix_uchar)
  SET_OPERAND(vcl::implicit_matrix_base<short>*,          implicit_matrix_short)
  SET_OPERAND(vcl::implicit_matrix_base<unsigned short>*, implicit_matrix_ushort)
  SET_OPERAND(vcl::implicit_matrix_base<int>*,            implicit_matrix_int)
  SET_OPERAND(vcl::implicit_matrix_base<unsigned int>*,   implicit_matrix_uint)
  SET_OPERAND(vcl::implicit_matrix_base<long>*,           implicit_matrix_long)
  SET_OPERAND(vcl::implicit_matrix_base<unsigned long>*,  implicit_matrix_ulong)
  SET_OPERAND(vcl::implicit_matrix_base<float>*,          implicit_matrix_float)
  SET_OPERAND(vcl::implicit_matrix_base<double>*,         implicit_matrix_double)

  SET_OPERAND(vcl::compressed_matrix<float>*,  compressed_matrix_float)
  SET_OPERAND(vcl::compressed_matrix<double>*, compressed_matrix_double)

  SET_OPERAND(vcl::coordinate_matrix<float>*,  coordinate_matrix_float)
  SET_OPERAND(vcl::coordinate_matrix<double>*, coordinate_matrix_double)

  SET_OPERAND(vcl::ell_matrix<float>*,  ell_matrix_float)
  SET_OPERAND(vcl::ell_matrix<double>*, ell_matrix_double)

  SET_OPERAND(vcl::hyb_matrix<float>*,  hyb_matrix_float)
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

  vcl::scheduler::statement get_vcl_statement() const {
    return vcl::scheduler::statement(vcl_expression_nodes);
  }

  void print_vcl_statement() const {
    std::cout << get_vcl_statement() << std::endl;
  }

  void execute() {
    vcl::scheduler::execute(get_vcl_statement());
  }

#ifdef VIENNACL_WITH_OPENCL
  int check_template(atidlas::template_base const & tplt, viennacl::ocl::context const & context)
  {
    vcl::scheduler::statement tmp_statement(vcl_expression_nodes);
    return tplt.check_invalid(tmp_statement, context.current_device());
  }
  
  void execute_template(atidlas::template_base const & T, vcl::ocl::context & ctx, bool force_compilation)
  {
    atidlas::execute(T, get_vcl_statement(), ctx, force_compilation);
  }
#endif

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
