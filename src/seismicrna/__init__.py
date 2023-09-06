"""

SEISMIC-RNA Initialization Module

========================================================================

Expose the sub-packages demult, align, relate, cluster, and table, plus
the __version__ attribute, at the top level so that they can be imported
from external modules and scripts in either of the following manners:

>>> import seismicrna
>>> seismicrna.__version__

or

>>> from seismicrna import __version__

"""


import warnings

from . import demult, align, relate, cluster, table, fastc

warnings.simplefilter(action='ignore', category=FutureWarning)

__version__ = "0.8.2"


########################################################################
#                                                                      #
# ©2023, the Rouskin Lab.                                              #
#                                                                      #
# This file is part of SEISMIC-RNA.                                    #
#                                                                      #
# SEISMIC-RNA is free software; you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation; either version 3 of the License, or    #
# (at your option) any later version.                                  #
#                                                                      #
# SEISMIC-RNA is distributed in the hope that it will be useful, but   #
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANT- #
# ABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General     #
# Public License for more details.                                     #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with SEISMIC-RNA; if not, see <https://www.gnu.org/licenses>.  #
#                                                                      #
########################################################################
