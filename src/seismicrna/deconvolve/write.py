from datetime import datetime
from math import inf
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import DeconvolveMutsDataset
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
from ..mask.data import load_mask_dataset
from ..mask.report import MaskReport


SEED_DTYPE = np.uint32

from .table import DeconvolvePosTable

def deconvolve(mask_report_file: Path,
            positions: Iterable[Iterable[int]],
            pattern,
            confidences,
            no_probe_sample: str,
            only_probe_sample: str,
            *,
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

    # print(path.build(DeconvolvePosTable.path_segs()))
    if need_write(deconvolve_report_file, force):
        began = datetime.now()
        # Load the unique reads.
        dataset = load_mask_dataset(mask_report_file)
        if no_probe_sample is not None and only_probe_sample is not None:
            # no_probe_report_file = recast_file_path(mask_report_file,
            #                                         MaskReport,
            #                                         MaskReport, 
            #                                         sample=no_probe_sample)
            # no_probe_dataset = load_mask_dataset(no_probe_report_file)
            # only_probe_report_file = recast_file_path(mask_report_file,
            #                                         MaskReport,
            #                                         MaskReport, 
            #                                         sample=only_probe_sample)
            # only_probe_dataset = load_mask_dataset(only_probe_report_file)
            pass
        else:
            no_probe_dataset = None
            only_probe_dataset = None
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
        report = DeconvolveReport.from_deconv_run(deconv_run,
                                                  began=began,
                                                  ended=ended,
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
