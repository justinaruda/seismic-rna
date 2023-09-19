from __future__ import annotations
from logging import getLogger

from ..core import path
from ..core.cmd import CMD_ALIGN
from ..core.report import Report

logger = getLogger(__name__)

BATCH_INDEX_COL = "Read Name"


class AlignReport(Report):
    __slots__ = ("sample",
                 "demultiplexed", "paired_end", "phred_enc",
                 "fastqc",
                 "cut",
                 "cut_q1", "cut_q2", "cut_g1", "cut_a1", "cut_g2", "cut_a2",
                 "cut_o", "cut_e", "cut_indels", "cut_nextseq",
                 "cut_discard_trimmed", "cut_discard_untrimmed", "cut_m",
                 "bt2_local",
                 "bt2_discordant", "bt2_mixed", "bt2_dovetail", "bt2_contain",
                 "bt2_unal", "bt2_score_min",
                 "bt2_i", "bt2_x", "bt2_gbar", "bt2_l", "bt2_s",
                 "bt2_d", "bt2_r", "bt2_dpad", "bt2_orient",
                 "min_mapq",
                 "reads_init", "reads_trim", "reads_align", "reads_filter",
                 "reads_refs")

    @classmethod
    def path_segs(cls):
        return path.SampSeg, path.CmdSeg, path.AlignRepSeg

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: CMD_ALIGN}

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
