from ..core import path
from ..core.cmd import CMD_MASK
from ..core.report import BatchReport


class MaskReport(BatchReport):
    __slots__ = (
        # Sample, reference, and section information.
        "sample", "ref", "sect", "end5", "end3",
        # Batch information.
        "checksums", "n_batches",
        # Types of mutations and matches to count.
        "count_muts", "count_refs",
        # Position filtering parameters.
        "exclude_gu", "exclude_polya", "exclude_pos",
        "min_ninfo_pos", "max_fmut_pos",
        # Position filtering results.
        "n_pos_init",
        "n_pos_gu", "n_pos_polya", "n_pos_user",
        "n_pos_min_ninfo", "n_pos_max_fmut", "n_pos_kept",
        "pos_gu", "pos_polya", "pos_user",
        "pos_min_ninfo", "pos_max_fmut", "pos_kept",
        # Read filtering parameters.
        "min_finfo_read", "max_fmut_read", "min_mut_gap",
        # Read filtering results.
        "n_reads_init",
        "n_reads_min_finfo", "n_reads_max_fmut", "n_reads_min_gap",
        "n_reads_kept",
    )

    @classmethod
    def path_segs(cls):
        return (path.SampSeg, path.CmdSeg, path.RefSeg, path.SectSeg,
                path.MaskRepSeg)

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: CMD_MASK}

    @classmethod
    def get_batch_seg(cls):
        return path.MaskBatSeg

########################################################################
#                                                                      #
# Copyright ©2023, the Rouskin Lab.                                              #
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
