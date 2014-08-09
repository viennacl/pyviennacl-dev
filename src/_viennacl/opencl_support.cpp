#ifdef VIENNACL_WITH_OPENCL

#include "vector.hpp"

#include <viennacl/ocl/backend.hpp>
#include <viennacl/ocl/command_queue.hpp>
#include <viennacl/ocl/context.hpp>
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/device_utils.hpp>
#include <viennacl/ocl/platform.hpp>
#include <viennacl/ocl/program.hpp>

template<class VCLType, class PtrType>
vcl::tools::shared_ptr<VCLType>
vcl_object_from_int_ptr(vcl::vcl_size_t int_ptr) {
  VCLType *p = new VCLType((PtrType)int_ptr);
  return vcl::tools::shared_ptr<VCLType>(p);
}

vcl::tools::shared_ptr<vcl::ocl::context>
vcl_context_from_int_ptr(vcl::vcl_size_t int_ptr) {
  vcl::ocl::context *c = new vcl::ocl::context();
  c->init((cl_context)int_ptr);
  return vcl::tools::shared_ptr<vcl::ocl::context>(c);
}

vcl::tools::shared_ptr<vcl::ocl::command_queue>
vcl_command_queue_from_int_ptr(const vcl::ocl::context& ctx, vcl::vcl_size_t int_ptr) {
  vcl::ocl::handle<cl_command_queue> h((cl_command_queue)int_ptr, ctx);
  vcl::ocl::command_queue *q = new vcl::ocl::command_queue(h);
  return vcl::tools::shared_ptr<vcl::ocl::command_queue>(q);
}

vcl::tools::shared_ptr<vcl::ocl::program>
vcl_program_from_int_ptr(const vcl::ocl::context ctx,
                         vcl::vcl_size_t int_ptr, std::string const& name)
{
  vcl::ocl::program *p = new vcl::ocl::program((cl_program)int_ptr, ctx, name);
  return vcl::tools::shared_ptr<vcl::ocl::program>(p);
}

template <class oclT>
vcl::vcl_size_t get_ocl_id(oclT& o) {
  return (vcl::vcl_size_t) o.id();
}

template <class oclT>
vcl::vcl_size_t get_ocl_ptr(oclT& o) {
  return (vcl::vcl_size_t) o.handle().get();
}

std::vector<vcl::ocl::device>
get_platform_devices(vcl::ocl::platform& p) {
  return p.devices();
}

std::string get_device_info(vcl::ocl::device& d) {
  return d.info();
}

std::string get_device_full_info(vcl::ocl::device& d) {
  return d.full_info();
}

std::string ctx_get_cache_path(vcl::ocl::context& ctx) {
  return std::string(ctx.cache_path());
}

void ctx_set_cache_path(vcl::ocl::context& ctx, const std::string& path) {
  ctx.cache_path(const_cast<char*>(path.c_str()));
}

const vcl::ocl::command_queue& ctx_get_queue(vcl::ocl::context& ctx,
                                             vcl::vcl_size_t device_int_ptr,
                                             vcl::vcl_size_t id) {
  return ctx.get_queue((cl_device_id)device_int_ptr, id);
}

void ctx_add_new_queue(vcl::ocl::context& ctx, vcl::vcl_size_t device) {
  ctx.add_queue((cl_device_id)device);
}

void ctx_add_existing_queue(vcl::ocl::context& ctx,
                            vcl::vcl_size_t device, vcl::vcl_size_t queue) {
  ctx.add_queue((cl_device_id)device, (cl_command_queue)queue);
}

void ctx_add_program(vcl::ocl::context& ctx,
                     vcl::vcl_size_t program,
                     std::string const& name)
{
  ctx.add_program((cl_program)program, name);
}

bp::list ctx_get_programs(vcl::ocl::context& ctx)
{
  return std_vector_to_list<vcl::ocl::program>
    (ctx.get_programs());
}

#endif

PYVCL_SUBMODULE(opencl_support)
{

#ifdef VIENNACL_WITH_OPENCL
  PYTHON_SCOPE_SUBMODULE(opencl_support);

  bp::enum_<vcl::ocl::vendor_id>
            ("vendor_id")
            ENUM_VALUE(vcl::ocl, beignet_id)
            ENUM_VALUE(vcl::ocl, intel_id)
            ENUM_VALUE(vcl::ocl, nvidia_id)
            ENUM_VALUE(vcl::ocl, amd_id)
            ENUM_VALUE(vcl::ocl, unknown_id)
            ;

  bp::enum_<vcl::ocl::device_architecture_family>
            ("device_architecture_family")
            ENUM_VALUE(vcl::ocl, tesla)
            ENUM_VALUE(vcl::ocl, fermi)
            ENUM_VALUE(vcl::ocl, kepler)
            
            ENUM_VALUE(vcl::ocl, evergreen)
            ENUM_VALUE(vcl::ocl, northern_islands)
            ENUM_VALUE(vcl::ocl, southern_islands)
            ENUM_VALUE(vcl::ocl, volcanic_islands)
            
            ENUM_VALUE(vcl::ocl, unknown)
            ;
          
  bp::class_<vcl::ocl::platform, vcl::tools::shared_ptr<vcl::ocl::platform> >
    ("platform", bp::no_init)
    .def("__init__", bp::make_constructor(vcl_object_from_int_ptr<vcl::ocl::platform, cl_platform_id>))
    .add_property("info", &vcl::ocl::platform::info)
    .add_property("devices", get_platform_devices)
    .add_property("int_ptr", get_ocl_id<vcl::ocl::platform>)
    ;

  bp::to_python_converter<std::vector<vcl::ocl::platform>,
                          vector_to_list_converter<vcl::ocl::platform> >();

  bp::def("get_platforms", vcl::ocl::get_platforms);
  bp::def("get_architecture_family", vcl::ocl::get_architecture_family);

  bp::class_<vcl::ocl::device, vcl::tools::shared_ptr<vcl::ocl::device> >
    ("device", bp::no_init)
    .def("__init__", bp::make_constructor(vcl_object_from_int_ptr<vcl::ocl::device, cl_device_id>))
    .add_property("name", &vcl::ocl::device::name)
    .add_property("vendor", &vcl::ocl::device::vendor)
    .add_property("version", &vcl::ocl::device::version)
    .add_property("driver_version", &vcl::ocl::device::driver_version)
    .add_property("info", get_device_info)
    .add_property("full_info", get_device_full_info)
    .add_property("extensions", &vcl::ocl::device::extensions)
    .add_property("double_support", &vcl::ocl::device::double_support)
    .add_property("int_ptr", get_ocl_id<vcl::ocl::device>)
    .add_property("architecture_family", &vcl::ocl::device::architecture_family)
    ;

  bp::to_python_converter<std::vector<vcl::ocl::device>,
                          vector_to_list_converter<vcl::ocl::device> >();
  
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  init, init_new_context,
                                  ());
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  add_device, context_add_device,
                                  (viennacl::ocl::device const&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  switch_device, context_switch_device,
                                  (viennacl::ocl::device const&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  platform_index, context_set_platform_index,
                                  (vcl::vcl_size_t));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, vcl::vcl_size_t,
                                  platform_index, context_get_platform_index,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  switch_queue, ctx_switch_queue,
                                  (const vcl::ocl::command_queue&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context,
                                  vcl::ocl::program&,
                                  get_program,
                                  ctx_get_program_from_string,
                                  (const std::string&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context,
                                  vcl::ocl::program&,
                                  get_program,
                                  ctx_get_program_from_int,
                                  (vcl::vcl_size_t));
  bp::class_<vcl::ocl::context>("context", bp::no_init)
    .def("__init__", bp::make_constructor(vcl_context_from_int_ptr))
    .def("init_new_context", init_new_context)
    .add_property("cache_path", ctx_get_cache_path, ctx_set_cache_path)
    .add_property("current_device",
                  bp::make_function(&vcl::ocl::context::current_device,
                                    bp::return_value_policy<bp::reference_existing_object>()))
    .add_property("devices",
                  bp::make_function(&vcl::ocl::context::devices,
                                    bp::return_value_policy<bp::reference_existing_object>()))
    .def("add_device", context_add_device)
    .def("switch_active_device", context_switch_device)
    .add_property("current_queue",
                  bp::make_function(&vcl::ocl::context::current_queue,
                                    bp::return_value_policy<bp::reference_existing_object>()))
    .def("get_queue", ctx_get_queue, bp::return_value_policy<bp::reference_existing_object>())
    .def("add_new_queue", ctx_add_new_queue)
    .def("add_existing_queue", ctx_add_existing_queue)
    .def("switch_queue", ctx_switch_queue)
    .def("add_program", ctx_add_program)
    .def("has_program", &vcl::ocl::context::has_program)
    .def("get_program_from_string", ctx_get_program_from_string,
         bp::return_value_policy<bp::copy_non_const_reference>())
    .def("get_program_from_int", ctx_get_program_from_int,
         bp::return_value_policy<bp::copy_non_const_reference>())
    .def("get_programs", ctx_get_programs)
    .add_property("program_num", &vcl::ocl::context::program_num)
    .def("delete_program", &vcl::ocl::context::delete_program)
    .add_property("platform_index", context_get_platform_index, context_set_platform_index)
    .add_property("int_ptr", get_ocl_ptr<vcl::ocl::context>)
    ;

  bp::class_<vcl::ocl::command_queue>("command_queue", bp::no_init)
    .def("__init__", bp::make_constructor(vcl_command_queue_from_int_ptr))
    .add_property("int_ptr", get_ocl_ptr<vcl::ocl::command_queue>)
    .def("finish", &vcl::ocl::command_queue::finish)
    ;
    
  DISAMBIGUATE_FUNCTION_PTR(void,
                            vcl::ocl::setup_context,
                            setup_context_single,
                            (long, vcl::ocl::device const&));
  bp::def("setup_context", setup_context_single);

  bp::def("switch_context", vcl::ocl::switch_context);

 bp::class_<vcl::ocl::program, vcl::tools::shared_ptr<vcl::ocl::program> >
   ("program", bp::no_init)
   .def("__init__", bp::make_constructor(vcl_program_from_int_ptr))
   .add_property("int_ptr", get_ocl_ptr<vcl::ocl::program>)
   .add_property("name",
                 bp::make_function(&vcl::ocl::program::name,
                                   bp::return_value_policy<bp::reference_existing_object>()))
   ;
 
#endif

}
