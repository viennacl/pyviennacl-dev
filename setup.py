#!/usr/bin/env python
import sys
from distutils.command.build_ext import build_ext

platform_cflags = {}
platform_ldflags = {}
platform_libs = {}


class build_ext_subclass(build_ext):
    """Shamelessly stolen from
    https://stackoverflow.com/questions/724664
    """
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in platform_cflags.keys():
            for e in self.extensions:
                e.extra_compile_args = platform_cflags[c]
        if c in platform_ldflags.keys():
            for e in self.extensions:
                e.extra_link_args = platform_ldflags[c]
        if c in platform_libs.keys():
            for e in self.extensions:
                try:
                    e.libraries += platform_libs[c]
                except:
                    e.libraries = platform_libs[c]
        build_ext.build_extensions(self)


def get_config_schema():
    from aksetup_helper import ConfigSchema, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    if sys.platform.startswith('darwin'):
        import platform
        osx_ver, _, _ = platform.mac_ver()
        osx_ver = '.'.join(osx_ver.split('.')[:2])

        sysroot_paths = [
                "/Applications/Xcode.app/Contents/Developer/Platforms/"
                "MacOSX.platform/Developer/SDKs/MacOSX%s.sdk" % osx_ver,
                "/Developer/SDKs/MacOSX%s.sdk" % osx_ver
                ]

        default_cl_libs = []
        default_cxxflags = ['-arch', 'i386', '-arch', 'x86_64']

        from os.path import isdir
        for srp in sysroot_paths:
            if isdir(srp):
                default_cxxflags.extend(['-isysroot', srp])
                break

        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]

    else:
        default_cl_libs = ["OpenCL"]
        default_cxxflags = []
        default_ldflags = []

    try:
        import os
        conf_file = os.environ["SITECONF"]
    except:
        conf_file = "siteconf.py"

    return ConfigSchema(
        make_boost_base_options() + [
            BoostLibraries("python"),

            Switch("DEBUG", False, "Turn on debug mode"),

            Switch("USE_SHIPPED_BOOST", True, "Use included Boost library"),
            Switch("USE_SHIPPED_VIENNACL", True, "Use included ViennaCL library"),
            
            Switch("USE_OPENCL", True, "Use OpenCL"),
            IncludeDir("CL", []),
            LibraryDir("CL", []),
            Libraries("CL", default_cl_libs),

            Switch("USE_OPENMP", True, "Use OpenMP"),

            IncludeDir("VIENNACL", []),

            StringListOption("CXXFLAGS", default_cxxflags,
                             help="Any extra C++ compiler options to include"),
            StringListOption("LDFLAGS", default_ldflags,
                             help="Any extra linker options to include"),
        ],
        conf_file=conf_file)


def main():
    import os
    from aksetup_helper import (hack_distutils, get_config, setup,
            NumpyExtension, set_up_shipped_boost_if_requested,
            check_git_submodules)

    #check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema(),
            warn_about_no_config=False)

    EXTRA_OBJECTS, EXTRA_DEFINES = \
            set_up_shipped_boost_if_requested(
                    "pyviennacl", conf,
                    source_path="external/boost-python-ublas-subset/boost_subset",
                    boost_chrono="header_only")

    INCLUDE_DIRS = ["external/boost_numpy/"]
    INCLUDE_DIRS += conf["BOOST_INC_DIR"] # This is set by set_up_shipped_boost_if_requested

    if conf["USE_SHIPPED_VIENNACL"]:
        INCLUDE_DIRS += ["external/viennacl-dev/"]
    else:
        INCLUDE_DIRS += conf["VIENNACL_INC_DIR"]

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    # {{{ get version number

    ver_dic = {}
    version_file = open("pyviennacl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyviennacl/version.py", 'exec'), ver_dic)

    # }}}

    platform_cflags["msvc"] = ["/EHsc"]
    platform_libs["msvc"] = []
    platform_cflags["unix"] = ["-Wno-unused-function",
                               "-Wno-unused-local-typedefs"]
    platform_libs["unix"] = []
    platform_cflags["mingw32"] = platform_cflags["unix"]
    platform_libs["mingw32"] = []

    if conf["USE_OPENCL"]:
        EXTRA_DEFINES["VIENNACL_WITH_OPENCL"] = None

    if conf["USE_OPENMP"]:
        EXTRA_DEFINES["VIENNACL_WITH_OPENMP"] = None
        platform_libs["msvc"] += ["/openmp"]
        if sys.platform.startswith("darwin"):
            print("-----------------------------------------------------------")
            print("NB: OpenMP not currently supported with the clang compiler!")
            print("-----------------------------------------------------------")
        else:
            platform_cflags["unix"] += ["-fopenmp"]
            platform_libs["unix"] += ["gomp"]

    EXTRA_DEFINES["VIENNACL_WITH_UBLAS"] = None

    UNDEF_MACROS = []
    if conf["DEBUG"]:
         UNDEF_MACROS += ['NDEBUG']
         EXTRA_DEFINES["VIENNACL_DEBUG_ALL"] = None
         EXTRA_DEFINES["VIENNACL_NO_CURRENT_CONTEXT"] = None
         #EXTRA_DEFINES["VIENNACL_BUILD_INFO"] = None
         platform_cflags["unix"] += ["-O0","-ggdb"]

    if not sys.platform.startswith("darwin"):
        platform_libs['unix'] += ['rt']

    source_files = [
        "core", "scalars",
        "vector_float", "vector_double", "vector_int", "vector_long",
        "vector_uint", "vector_ulong",

        "dense_matrix_float", "dense_matrix_double",
        "dense_matrix_int", "dense_matrix_long",
        "dense_matrix_uint", "dense_matrix_ulong",

        "structured_matrices",

        "cpu_sparse_matrix",
        "compressed_matrix", "compressed_compressed_matrix",
        "ell_matrix", "sliced_ell_matrix",
        "coordinate_matrix", "hyb_matrix",

        "preconditioners","direct_solvers", "iterative_solvers",

        "extra_functions", "eig", "bandwidth_reduction",

        "scheduler", "platform_support", "opencl_support"]

    from glob import glob

    try:
        long_description = open("README.rst", "rt", encoding='utf8').read()
    except:
        long_description = open("README.rst", "rt").read()

    setup(
        name="pyviennacl",
        version=ver_dic["VERSION_TEXT"],
        description="Sparse/dense linear algebra on GPUs and CPUs using OpenCL and OpenMP",
        long_description=long_description,
        author='Toby St Clere Smithe',
        author_email='pyviennacl@tsmithe.net',
        url="http://viennacl.sourceforge.net/pyviennacl.html",
        classifiers=[
            'Environment :: Console',
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: C++',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
        ],
        setup_requires=['numpy>=1.6'],
        install_requires=['numpy>=1.6', 'appdirs'],
        tests_require=['pytest'],

        packages=["pyviennacl"],
        ext_package="pyviennacl",
        ext_modules=[NumpyExtension(
            "_viennacl",

            [os.path.join("src", "_viennacl", sf + ".cpp")
             for sf in source_files]
            + glob("external/boost_numpy/libs/numpy/src/*.cpp")
            + EXTRA_OBJECTS,
            depends=[os.path.join("src", "_viennacl", "viennacl.h")],

            extra_compile_args=conf["CXXFLAGS"],
            extra_link_args=['-Wl,-soname=_viennacl.so'] + conf["LDFLAGS"],

            define_macros=list(EXTRA_DEFINES.items()),
            undef_macros=UNDEF_MACROS,

            include_dirs=INCLUDE_DIRS,
            library_dirs=LIBRARY_DIRS + conf["CL_LIB_DIR"],
            libraries=LIBRARIES + conf["CL_LIBNAME"],
        )],
        cmdclass={'build_ext': build_ext_subclass}
    )


if __name__ == "__main__":
    main()
