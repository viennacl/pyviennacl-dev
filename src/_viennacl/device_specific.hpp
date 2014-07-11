#ifndef VIENNACL_DEVICE_SPECIFIC_H

#include <viennacl/device_specific/templates/vector_axpy_template.hpp>
#include <viennacl/device_specific/templates/matrix_axpy_template.hpp>
#include <viennacl/device_specific/templates/reduction_template.hpp>
#include <viennacl/device_specific/templates/row_wise_reduction_template.hpp>
#include <viennacl/device_specific/templates/matrix_product_template.hpp>

#include "scheduler.hpp"

namespace ds = viennacl::device_specific;

vcl::tools::shared_ptr<ds::statements_container>
make_statements_tuple(bp::list statement_wrappers,
                      ds::statements_container::order_type order) {

  std::list<vcl::scheduler::statement> statements;
  
  for (long i=0; i < bp::len(statement_wrappers); ++i) {
    const statement_wrapper& w = bp::extract<statement_wrapper>(statement_wrappers);
    statements.push_front(w.get_vcl_statement());
  }
 
  ds::statements_container* s = new ds::statements_container(statements, order);
  
  return vcl::tools::shared_ptr<ds::statements_container>(s);
}

#endif
