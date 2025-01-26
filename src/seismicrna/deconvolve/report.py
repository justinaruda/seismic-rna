from datetime import datetime

from ..core import path
from .io import DeconvolveIO, DeconvolveBatchIO
from typing import Any
from ..core.report import (Field,
                           BatchedReport,
                           SampleF,
                           RefF,
                           RegF,
                           DeconvolveMutsF,
                           DeconvolveRefsF,
                           DeconvolveClusterMappingF,
                           DeconvolveMinReadsF,
                           DeconvolveReadCountsF,
                           DeconvolveClusterCountF,
                           DeconvolveConfidenceThreshF,
                           DeconvolveConfidenceF,
                           DeconvolveNoProbeSampleF,
                           DeconvolveOnlyProbeSampleF)


class DeconvolveReport(BatchedReport, DeconvolveIO):
        
    @classmethod
    def file_seg_type(cls):
        return path.DeconvRepSeg
    @classmethod
    def _batch_types(cls):
        return DeconvolveBatchIO,
    
    @classmethod
    def fields(cls):
        return [
            # Sample, reference, and region information.
            SampleF,
            RefF,
            RegF,
            # Clustering parameters.
            DeconvolveMutsF,
            DeconvolveRefsF,
            DeconvolveClusterMappingF,
            DeconvolveMinReadsF,
            DeconvolveReadCountsF,
            DeconvolveClusterCountF,
            DeconvolveConfidenceThreshF,
            DeconvolveConfidenceF,
            DeconvolveNoProbeSampleF,
            DeconvolveOnlyProbeSampleF,
        ] + super().fields()

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: path.CMD_DECONV_DIR}
    
    @classmethod
    def from_deconv_run(cls, deconv_run,
                        conf_thresh: float,
                        deconv_confs: dict[str,float],
                        no_probe_sample: str,
                        only_probe_sample: str,
                        began: datetime, 
                        ended:datetime, 
                        checksums: list[str]):
        """ Create a DeconvolveReport from a Deconvolver object. """
        # Initialize a new DeconvolveReport.
        return cls(sample=deconv_run.dataset.sample,
                   ref=deconv_run.dataset.ref,
                   reg=deconv_run.dataset.region.name,
                   n_batches=deconv_run.dataset.num_batches,
                   mut_pattern=deconv_run.pattern.yes,
                   ref_pattern=deconv_run.pattern.nos,
                   conf_thresh=conf_thresh,
                   deconv_confs=deconv_confs,
                   deconv_min_reads=deconv_run.min_reads,
                   no_probe_sample=no_probe_sample,
                   only_probe_sample=only_probe_sample,
                   checksums={DeconvolveBatchIO.btype(): checksums},
                   deconvolution_mapping=deconv_run.mapping_indexes,
                   deconvolve_read_counts=deconv_run.read_counts,
                   deconvolve_cluster_count=deconv_run.num_clusters,
                   began=began,
                   ended=ended)
    
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
