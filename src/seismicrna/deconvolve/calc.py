import pandas as pd

from ..core.rel import RelPattern
from ..mask.table import MaskPositionTable

def calc_bayes(no_probe_table: MaskPositionTable,
               only_probe_table: MaskPositionTable,
               pattern: RelPattern) -> pd.Series:

    refs = [ref for ref, muts in pattern.yes.patterns.items() if muts != 0]

    no_probe_mus = None
    only_probe_mus = None

    for mut_str in pattern.yes.to_report_format():
        ref, mut = mut_str.split(" -> ")

        if mut in 'ACGT':
            rel_label = f"Subbed-{mut}"
        elif mut == "D":
            rel_label = "Deleted"
        elif mut == "I":
            rel_label = "Inserted"
        else:
            continue

        np_ref = no_probe_table.fetch_ratio(rel=[rel_label], squeeze=True)
        mask_np = np_ref.index.get_level_values("Base") == ref
        np_ref = np_ref.where(mask_np, 0)

        op_ref = only_probe_table.fetch_ratio(rel=[rel_label], squeeze=True)
        mask_op = op_ref.index.get_level_values("Base") == ref
        op_ref = op_ref.where(mask_op, 0)

        if no_probe_mus is None:
            no_probe_mus = np_ref.copy()
        else:
            no_probe_mus = no_probe_mus.add(np_ref, fill_value=0)

        if only_probe_mus is None:
            only_probe_mus = op_ref.copy()
        else:
            only_probe_mus = only_probe_mus.add(op_ref, fill_value=0)

    if no_probe_mus is None or only_probe_mus is None:
        return pd.Series(dtype=float)

    denom = only_probe_mus.add(no_probe_mus)
    bayes = pd.Series(0.0, index=denom.index)
    nonzero = denom != 0
    bayes.loc[nonzero] = 0.98 * no_probe_mus[nonzero].divide(denom[nonzero])

    keep = bayes.index.get_level_values("Base").isin(refs)
    return bayes.loc[keep]

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
