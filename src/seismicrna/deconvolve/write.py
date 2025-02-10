from datetime import datetime
from math import inf
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import DeconvolveMutsDataset
from .deconv import DeconvRun
from .io import DeconvolveBatchWriter
from .report import DeconvolveReport
from .table import DeconvolveDatasetTabulator
from ..cluster.uniq import UniqReads
from ..core import path
from ..core.header import validate_ks
from ..core.io import recast_file_path
from ..core.logs import logger
from ..core.task import dispatch
from ..core.tmp import release_to_out
from ..core.types import get_max_uint
from ..core.write import need_write
from ..mask.dataset import load_mask_dataset
from ..mask.report import MaskReport


SEED_DTYPE = np.uint32

from .table import DeconvolvePositionTable

def deconvolve(mask_report_file: Path,
            positions: Iterable[Iterable[int]],
            pattern,
            confidences,
            no_probe_sample: str,
            only_probe_sample: str,
            *,
            corr_editing_bias: bool = False,
            conf_thresh: bool,
            tmp_dir: Path,
            n_procs: int,
            brotli_level: int,
            force: bool,
            deconvolve_pos_table: bool,
            deconvolve_abundance_table: bool,
            strict: bool = False,
            **kwargs):
    norm_edits = kwargs.pop("norm_edits", None)
    """ Deconvolve unique reads from one mask dataset. """
    # Check if the deconvolve report file already exists.
    deconvolve_report_file = recast_file_path(mask_report_file,
                                           MaskReport,
                                           DeconvolveReport)

    if need_write(deconvolve_report_file, force):
        began = datetime.now()
        # Load the unique reads.
        dataset = load_mask_dataset(mask_report_file)
        deconv_run = DeconvRun(dataset=dataset,
                               positions=positions,
                               pattern=pattern,
                               norm_edits=norm_edits,
                               strict=strict)
        # Output the deconvolve memberships in batches of reads.
        batch_writer = DeconvolveBatchWriter(deconv_run,
                                             brotli_level,
                                             tmp_dir)
        batch_writer.write_batches()
        # Write the deconvolve report.
        ended = datetime.now()
        pos_strings = [f"edited_{'_'.join(str(pos) for pos in position)}" for position in positions]
        pos_confs = [
            confidences.loc[pos].iloc[0] if len(pos) == 1 else
            confidences.loc[list(pos)].prod()
            for pos in positions]
        deconv_confs = {pos_string: confidence for pos_string, confidence in zip(pos_strings, pos_confs)}
        report = DeconvolveReport.from_deconv_run(deconv_run,
                                                  began=began,
                                                  ended=ended,
                                                  conf_thresh=conf_thresh,
                                                  deconv_confs=deconv_confs,
                                                  no_probe_sample=no_probe_sample,
                                                  only_probe_sample=only_probe_sample,
                                                  checksums=batch_writer.checksums)
        report_saved = report.save(tmp_dir)
        release_to_out(dataset.top, tmp_dir, report_saved.parent)
    # Write the tables if they do not exist.
    DeconvolveDatasetTabulator(
        dataset=DeconvolveMutsDataset(deconvolve_report_file),
        count_pos=deconvolve_pos_table,
        count_read=False,
        positions_list=positions,
        confidences_list = confidences,
        corr_editing_bias = corr_editing_bias,
        max_procs=n_procs,
    ).write_tables(pos=deconvolve_pos_table, clust=deconvolve_abundance_table)
    return deconvolve_report_file.parent
    

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
