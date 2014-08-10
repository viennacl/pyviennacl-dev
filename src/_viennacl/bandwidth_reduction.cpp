#include "bandwidth_reduction.hpp"

PYVCL_SUBMODULE(bandwidth_reduction)
{

  bp::class_<vcl::cuthill_mckee_tag>("cuthill_mckee_tag");
  
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::advanced_cuthill_mckee_tag,
                                  double,
                                  starting_node_param,
                                  get_acm_starting_node_param,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::advanced_cuthill_mckee_tag,
                                  void,
                                  starting_node_param,
                                  set_acm_starting_node_param,
                                  (double))
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::advanced_cuthill_mckee_tag,
                                  vcl::vcl_size_t,
                                  max_root_nodes,
                                  get_acm_max_root_nodes,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::advanced_cuthill_mckee_tag,
                                  void,
                                  max_root_nodes,
                                  set_acm_max_root_nodes,
                                  (vcl::vcl_size_t))
  bp::class_<vcl::advanced_cuthill_mckee_tag>
    ("advanced_cuthill_mckee_tag",
     bp::init<double, vcl::vcl_size_t>())
    .add_property("starting_node_param",
                  get_acm_starting_node_param, set_acm_starting_node_param)
    .add_property("max_root_nodes",
                  get_acm_max_root_nodes, set_acm_max_root_nodes)
    ;

  bp::class_<vcl::gibbs_poole_stockmeyer_tag>("gibbs_poole_stockmeyer_tag");

}
