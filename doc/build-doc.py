import sys
from sphinx import main

sys.exit(main(['sphinx-build_pyviennacl', 'doc', 'build/doc']))

