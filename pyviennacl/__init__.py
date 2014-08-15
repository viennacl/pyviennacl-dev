"""PyViennaCL"""

from .version import VERSION_TEXT as __version__
from ._viennacl import __version__ as __viennacl_version__

from .pycore import *
from .linalg import *
from .tags import *
from .vclmath import *
from .backend import *
try: from .opencl import *
except ImportError: pass
from .io import *
from .util import *
from .device_specific import *

from . import vclmath as math

# TODO: __all__
# TODO: __copyright__
