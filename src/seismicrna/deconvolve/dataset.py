from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import pandas as pd

from .batch import DeconvolveMutsBatch
from .io import DeconvolveBatchIO
from .report import DeconvolveReport
from ..core.batch import MutsBatch
from ..core.data import (Dataset,
                         LoadedDataset,
                         LoadFunction,
                         MergedUnbiasDataset,
                         MultistepDataset,
                         UnbiasDataset)
from ..core.header import (NUM_CLUSTS_NAME,
                           ClustHeader,
                           list_clusts,
                           list_ks_clusts,
                           validate_ks)
from ..core.join.data import (BATCH_NUM,
                              READ_NUMS,
                              SEG_END5S,
                              SEG_END3S,
                              MUTS,
                              RESPS,
                              JoinMutsDataset)
from ..core.report import (DeconvolveMutsF,
                           DeconvolveRefsF,
                           DeconvolveClusterMappingF,
                           DeconvolveMinReadsF,
                           DeconvolveReadCountsF,
                           DeconvolveConfidenceThreshF,
                           DeconvolveConfidenceF,
                           DeconvolveClusterCountF,
                           DeconvolveNoProbeSampleF,
                           DeconvolveOnlyProbeSampleF)
from ..mask.batch import MaskMutsBatch
from ..mask.data import load_mask_dataset
from ..mask.report import MaskReport
from ..mask.table import MaskPositionTable

from ..core.rel import RelPattern
from ..core import path



class DeconvolveDataset(Dataset, ABC):
    """ Dataset for clustered data. """

    @cached_property
    @abstractmethod
    def name_to_cluster(self) -> dict[str:int]:
        """ Name to cluster mapping """
    
    @cached_property
    @abstractmethod
    def name_to_cluster_index(self):
        """ Name to cluster index mapping """
    
    @cached_property
    @abstractmethod
    def cluster_to_name(self) -> dict[int:str]:
        """ Cluster to name mapping """

    @cached_property
    @abstractmethod
    def cluster_index_to_name(self):
        """ Cluster index to name mapping """
    
    @cached_property
    @abstractmethod
    def read_counts(self) -> int:
        """ Number of reads per cluster. """
        
    @cached_property
    @abstractmethod
    def num_clusters(self) -> int:
        """ Number of clusters. """


class DeconvolveReadDataset(DeconvolveDataset, LoadedDataset):
    """ Load clustering results. """

    @classmethod
    def get_report_type(cls):
        return DeconvolveReport

    @classmethod
    def get_batch_type(cls):
        return DeconvolveBatchIO

    @cached_property
    def deconv_pattern(self):
        return RelPattern(self.report.get_field(DeconvolveMutsF),
                          self.report.get_field(DeconvolveRefsF))

    @cached_property
    def name_to_cluster(self):
        return self.report.get_field(DeconvolveClusterMappingF)
    
    @cached_property
    def name_to_cluster_index(self):
        return {k:(self.num_clusters, v+1) 
                for k, v in self.name_to_cluster.items()}
    
    @cached_property
    def cluster_to_name(self):
        return {v: k for k, v in self.name_to_cluster.items()}
    
    @cached_property
    def cluster_index_to_name(self):
        return {v: k for k, v in self.name_to_cluster_index.items()}

    @cached_property
    def deconv_min_reads(self):
        return self.report.get_field(DeconvolveMinReadsF)

    @cached_property
    def read_counts(self):
        return self.report.get_field(DeconvolveReadCountsF)

    @cached_property
    def conf_thresh(self):
        return self.report.get_field(DeconvolveConfidenceThreshF)

    @cached_property
    def deconv_confs(self):
        return self.report.get_field(DeconvolveConfidenceF)

    @cached_property
    def num_clusters(self):
        return self.report.get_field(DeconvolveClusterCountF)
    
    @cached_property
    def no_probe_sample(self):
        return self.report.get_field(DeconvolveNoProbeSampleF)
    
    @cached_property
    def no_probe_path(self):
        return MaskPositionTable.build_path(top=self.top, 
                                     sample=self.no_probe_sample, 
                                     ref=self.ref,
                                     reg=self.region.name)
    
    @cached_property
    def only_probe_sample(self):
        return self.report.get_field(DeconvolveOnlyProbeSampleF)
    
    @cached_property
    def only_probe_path(self):
        return MaskPositionTable.build_path(top=self.top, 
                                     sample=self.only_probe_sample, 
                                     ref=self.ref,
                                     reg=self.region.name)

    @property
    def pattern(self):
        return None


class DeconvolveMutsDataset(DeconvolveDataset, MultistepDataset, UnbiasDataset):
    """ Merge cluster responsibilities with mutation data. """

    @classmethod
    def get_dataset1_load_func(cls):
        return load_mask_dataset

    @classmethod
    def get_dataset2_type(cls):
        return DeconvolveReadDataset

    @property
    def pattern(self):
        return self.data1.pattern

    @pattern.setter
    def pattern(self, pattern):
        self.data1.pattern = pattern

    @property
    def region(self):
        return self.data1.region

    @property
    def min_mut_gap(self):
        return getattr(self.data1, "min_mut_gap")

    @min_mut_gap.setter
    def min_mut_gap(self, min_mut_gap):
        self.data1.min_mut_gap = min_mut_gap

    @property
    def quick_unbias(self):
        return getattr(self.data1, "quick_unbias")

    @property
    def quick_unbias_thresh(self):
        return getattr(self.data1, "quick_unbias_thresh")

    @property
    def deconv_pattern(self):
        return getattr(self.data2, "deconv_pattern")

    @property
    def name_to_cluster(self):
        return getattr(self.data2, "name_to_cluster")

    @property
    def name_to_cluster_index(self):
        return getattr(self.data2, "name_to_cluster_index")
    
    @property
    def cluster_to_name(self):
        return getattr(self.data2, "cluster_to_name")

    @property
    def cluster_index_to_name(self):
        return getattr(self.data2, "cluster_index_to_name")
    
    @cached_property
    def deconv_min_reads(self):
        return getattr(self.data2, "deconv_min_reads")

    @property
    def read_counts(self):
        return getattr(self.data2, "read_counts")
    
    @property
    def conf_thresh(self):
        return getattr(self.data2, "conf_thresh")

    @property
    def deconv_confs(self):
        return getattr(self.data2, "deconv_confs")
    
    @property
    def num_clusters(self):
        return getattr(self.data2, "num_clusters")
    
    @property
    def no_probe_sample(self):
        return getattr(self.data2, "no_probe_sample")
    
    @property
    def no_probe_path(self):
        return getattr(self.data2, "no_probe_path")
    
    @property
    def only_probe_sample(self):
        return getattr(self.data2, "only_probe_sample")
    
    @property
    def only_probe_path(self):
        return getattr(self.data2, "only_probe_path")

    def _integrate(self, batch1: MaskMutsBatch, batch2: DeconvolveBatchIO):
        return DeconvolveMutsBatch(batch=batch1.batch,
                                region=batch1.region,
                                seg_end5s=batch1.seg_end5s,
                                seg_end3s=batch1.seg_end3s,
                                muts=batch1.muts,
                                resps=batch2.resps,
                                sanitize=False)


load_deconvolve_dataset = LoadFunction(DeconvolveMutsDataset)

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
