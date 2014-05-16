*****************
PyViennaCL Readme
*****************


PyViennaCL around the web
=========================

* `Home page <http://viennacl.sourceforge.net/pyviennacl.html>`_
* `Documentation <http://viennacl.sourceforge.net/pyviennacl/doc/index.html>`_
* `GitHub <https://github.com/viennacl/pyviennacl-dev>`_ (`Issue tracker <https://github.com/viennacl/pyviennacl-dev/issues>`_)


About PyViennaCL
================

*PyViennaCL* aims to make fast, powerful GPGPU and heterogeneous
scientific computing really transparently easy, especially for users
already using NumPy for representing matrices.

PyViennaCL does this by harnessing the `ViennaCL
<http://viennacl.sourceforge.net/>`_ linear algebra and numerical computation
library for GPGPU and heterogeneous systems, thereby making available to Python
programmers ViennaCL’s fast *OpenCL* and *CUDA* algorithms. PyViennaCL does
this in a way that is idiomatic and compatible with the Python community’s most
popular scientific packages, *NumPy* and *SciPy*.

PyViennaCL exposes the following functionality:

* sparse (compressed, co-ordinate, ELL, and hybrid) and dense
  (row-major and column-major) matrices, vectors and scalars on your
  compute device using OpenCL;
* standard arithmetic operations and mathematical functions;
* fast matrix products for sparse and dense matrices, and inner and
  outer products for vectors;
* direct solvers for dense triangular systems;
* iterative solvers for sparse and dense systems, using the BiCGStab,
  CG, and GMRES algorithms;
* iterative algorithms for eigenvalue estimation problems.


Requirements
============

PyViennaCL requires the following build dependencies:

* a C++ compiler;
* *Python* (2.7+) with *setuptools* [see below];
* *NumPy* (1.7+);
* *Sphinx* (1.0+) [if you want to build HTML documentation].

It is also highly recommended to have an available *OpenCL*
implementation, since PyViennaCL was designed for this case. If you do
not have OpenCL available, then PyViennaCL will still build and
function, but will not perform to full effect.

If you have a Boost installation on your system, then PyViennaCL can
also optionally use that, rather than the Boost subset shipped with
the PyViennaCL sources.

setuptools
----------

PyViennaCL uses the Python package `setuptools
<https://bitbucket.org/pypa/setuptools>`_ for installation. Unless you
are using **Python 3.4**, which comes with the required pieces, you
may need to install setuptools. The **best way to do so**, which will
also ensure you have the Python package manager *pip* installed, is to
follow `the instructions here
<http://www.pip-installer.org/en/latest/installing.html>`_.

OpenCL
------

By default, you will need an OpenCL implementation. PyViennaCL can
still build and function without one, but will not perform to full
effect.

For information on how to get an OpenCL implementation on Linux, see this
`howto <http://wiki.tiker.net/OpenCLHowTo>`_.
See also the installation section in the
`ViennaCL Manual <http://viennacl.sourceforge.net/viennacl-manual-current.pdf>`_
on installation hints for Windows.

If you have an OpenCL installation that you installed and is not
managed by your system's package manager, then it is likely you will
need to use the '--cl-lib-dir' configure option.

If you do not wish to use OpenCL, then follow the *Custom
installation* instructions below, and use the '--no-use-opencl'
configure option.

Extra requirements for building from source on Windows systems
--------------------------------------------------------------

In addition to the components required above, Windows builds require
the use of the Windows SDK version specific to the version of your
Python installation. For instance, when you run Python, if you see the
following::

  Python 2.7.6 (default, Nov 10 2013, 19:24:18) [MSC v.1500 32 bit (Intel)] on win32

then you are using a release of Python 2.7 and you are building for a
32-bit installation of that version. You must therefore choose the SDK
version and set the build environment accordingly.

If you are building for a standard distribution (such as described in
the example above) of Python 2.6, 2.7, 3.0, 3.1 or 3.2 (either 32- or
64-bit), you will need the `Microsoft Windows SDK v7.0 for Windows 7
and .NET Framework 3.5 SP1
<http://www.microsoft.com/en-us/download/details.aspx?id=18950>`_.

If you are building for a standard distribution of Python 3.3 or later
(32- or 64-bit), you will need the `Microsoft Windows SDK v7.1 for
Windows 7 and .NET Framework 4
<https://www.microsoft.com/en-us/download/details.aspx?id=8442>`_.

Having installed the correct SDK version, you will need to open an SDK
command prompt and issue the following commands, according the
architecture of your Python installation.

If you are building for a 32-bit Python installation, use::

  $> set DISTUTILS_USE_SDK=1
  $> setenv /x86 /release

If you are building for a 64-bit Python installation, use::

  $> set DISTUTILS_USE_SDK=1
  $> setenv /x64 /release

At this point, you will be able to follow the instructions detailed
below under *Installation*. Issue all commands from within the SDK
command window.

You might also be interested in the `Python Tools for Visual Studio
<https://pytools.codeplex.com/>`_, though these are not required
either for building or using PyViennaCL.


Installation
============

Please note that the commands below assume that the various programs
invoked are available in your system PATH; if they are not, please
provide the full path to the executable on invocation.

This is especially relevant on Windows systems, where Python is not by
default installed into the path. So, in this case, you might have to
run something like::

  $> C:\Python2.7\python -m pip install pyviennacl

instead of just::

  $> pip install pyviennacl

Quick installation using pip
----------------------------

The following command should usually be enough to install PyViennaCL,
as long as you have setuptools and pip installed (see above)::

  $> pip install pyviennacl

If you have a recent installation of Python (2.7 or 3.3+), and a
recent installation of pip (1.4+), then this should be painless.

Note that this command assumes you have OpenCL available, and if you
are on Linux, an OpenCL SDK.

If this does not succeed, follow the instructions below.

Custom installation
-------------------

These instructions assume a UNIX-like platform (such as a Linux
distribution or Mac OS X), but will work on a well-configured Windows
system with little modification.

If you are building from the release package, move on to the next paragraph.
If you are building from git, then before doing anything else,
navigate to your PyViennaCL folder and get the external submodules with::

  $> git submodule update --init

(On Windows you usually want to use the git bash for this)

If you are in a hurry, then the following should just work::

  $> python setup.py build
  $> sudo python setup.py install

Otherwise, read the configuration options, configure, build and
install::

  $> ./configure.py --help
  $> ./configure.py --some-options
  $> python setup.py build
  $> sudo python setup.py install

To build the HTML documentation using Sphinx, first install PyViennaCL
as above, and then run::

  $> cd doc
  $> make html

after which HTML documentation will be in doc/_build/html. Or you can
just access the documentation from within the Python interpreter after
installation::

  >>> import pyviennacl
  >>> help(pyviennacl)


Authors and Contact
===================

PyViennaCL was developed during the 2013 *Google Summer of Code* by 
`Toby St Clere Smithe <pyviennacl@tsmithe.net>`_.

ViennaCL was developed under the aegis of the *Institute for Microelectronics*
at the *Vienna University of Technology*.

For any technical questions related to ViennaCL, please use `our
mailing list <viennacl-support@lists.sourceforge.net>`_.


Licence
=======

(Py)ViennaCL is distributed under the MIT (X11) License. See the file
LICENSE in the source root directory.
