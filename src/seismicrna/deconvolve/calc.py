from ..core.rel import RelPattern
from ..mask.table import MaskPositionTable

def calc_bayes(no_probe_table: MaskPositionTable,
               only_probe_table: MaskPositionTable,
               pattern: RelPattern):
    refs = [ref for ref, muts in pattern.yes.patterns.items() if muts != 0]
    no_probe_mus = next(no_probe_table.iter_profiles()).data
    only_probe_mus = next(only_probe_table.iter_profiles()).data
    bayes = 0.98 * no_probe_mus/(only_probe_mus + no_probe_mus)
    return bayes[bayes.index.get_level_values("Base").isin(refs)]

########################################################################
#                                                                      #
# © Copyright 2024, the Rouskin Lab.                                   #
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
