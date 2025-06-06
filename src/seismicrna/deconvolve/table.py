from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable

import pandas as pd
import numpy as np

from ..core.seq import DNA
from .dataset import DeconvolveMutsDataset
from ..core import path
from ..core.header import (ClustHeader,
                           RelClustHeader,
                           make_header,
                           parse_header,
                           validate_k_clust,
                           DECONVOLVE_PREFIX,
                           DECONV_NAME)
from ..core.table import (AbundanceTable,
                          RelTypeTable,
                          PositionTableWriter,
                          AbundanceTableWriter,
                          PRECISION)
from ..mask.table import (PartialTable,
                          PartialPositionTable,
                          PartialTabulator,
                          PartialDatasetTabulator)
from ..relate.table import TableLoader, PositionTableLoader
from ..core.table.base import (MUTAT_REL,
                               INFOR_REL)
from ..core.logs import logger
from ..core.write import need_write
from ..mask.table import adjust_counts, MaskPositionTableLoader
from ..core.seq import Region
from ..core.rna import RNAProfile

from .report import DeconvolveReport

from .calc import calc_bayes

from .dataset import load_deconvolve_dataset


class DeconvolveTable(RelTypeTable, ABC):

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     cls.min_denom = 1000 #HARDCODED

    @classmethod
    def kind(cls):
        return path.DECONVOLVE_STEP

    @classmethod
    def header_type(cls):
        return RelClustHeader

    @cached_property
    def report_path(self):
        return DeconvolveReport.build_path(top=self.top, 
                                     sample=self.sample, 
                                     ref=self.ref,
                                     reg=self.reg)


class DeconvolvePositionTable(DeconvolveTable, PartialPositionTable, ABC):
    
    @abstractmethod
    def named_index(self) -> pd.MultiIndex:
        """ Return the index with names """

    def get_positions(self, k: int, clust: int):
        name = self.format_clust_name(k, clust)
        if "unedited" in name:
            return list()
        else:
            return [int(part) for part in name.split('_') if part.isdigit()]
    
    def format_clust_name(self, k: int, clust: int):
        """ Format a pair of k and cluster numbers into a name.

        Parameters
        ----------
        k: int
            Number of clusters
        clust: int
            Cluster number

        Returns
        -------
        str
            Name converted to deconvolved cluster string.
        """
        validate_k_clust(k, clust)
        name = self.named_index.get_level_values(DECONV_NAME)[clust-1]
        return f"{DECONVOLVE_PREFIX} {name}"
    
    def _iter_profiles(self, *,
                       regions: Iterable[Region] | None,
                       quantile: float,
                       rel: str,
                       k: int | None,
                       clust: int | None):
        """ Yield RNA mutational profiles from a table. """
        if regions is not None:
            regions = list(regions)
        else:
            regions = [self.region]
        for hk, hc in self.header.clusts:
            if (k is None or k == hk) and (clust is None or clust == hc):
                data_name = path.fill_whitespace(self.format_clust_name(hk, hc),
                                                 fill="-")
                positions = self.get_positions(hk, hc)
                for region in regions:
                    prof_region = region.copy()
                    seq_list = list(prof_region.seq)
                    for position in positions:
                        position -= region.end5
                        seq_list[position] = "G" #HARDCODED
                    prof_region.seq = DNA("".join(seq_list))
                    prof_region.mask_gu()
                    self.region.mask_gu() #HARDCODED
                    yield RNAProfile(region=prof_region,
                                     sample=self.sample,
                                     data_reg=self.reg,
                                     data_name=data_name,
                                     data=self.fetch_ratio(quantile=quantile,
                                                           rel=rel,
                                                           k=hk,
                                                           clust=hc,
                                                           squeeze=True,
                                                           exclude_masked=True))


class DeconvolveAbundanceTable(AbundanceTable, PartialTable, ABC):

    @classmethod
    def kind(cls):
        return path.DECONVOLVE_STEP

    @classmethod
    def header_type(cls):
        return ClustHeader

    @classmethod
    def index_depth(cls):
        return cls.header_depth()

    def _get_header(self):
        return parse_header(self.data.index)


class DeconvolvePositionTableWriter(PositionTableWriter, DeconvolvePositionTable):
    
    @cached_property
    def named_index(self):
        return self.tabulator._get_named_index(self.data.columns)
    
    def write(self, force: bool):
        """ Write the table's rounded data to the table's CSV file. """
        if need_write(self.path, force):
            orig_cols = self.data.columns
            self.data.columns = self.named_index
            self.data.round(decimals=PRECISION).to_csv(self.path)
            self.data.columns = orig_cols
            logger.routine(f"Wrote {self} to {self.path}")
        return self.path


class DeconvolveAbundanceTableWriter(AbundanceTableWriter,
                                     DeconvolveAbundanceTable):
    pass


class DeconvolvePositionTableLoader(PositionTableLoader, DeconvolvePositionTable):
    """ Load cluster data indexed by position. """    
    
    @cached_property
    def data(self) -> pd.DataFrame:
        data = pd.read_csv(self.path,
                           index_col=self.index_cols(),
                           header=[0,1])
        self._named_index = data.columns
        # Any numeric data in the header will be read as strings and
        # must be cast to integers using parse_header.
        header = parse_header(data.columns)
        # The columns must be replaced with the header index for the
        # type casting to take effect.
        data.columns = header.index
        return data

    @cached_property 
    def named_index(self):
        self.data
        return self._named_index


class DeconvolveAbundanceTableLoader(TableLoader, DeconvolveAbundanceTable):
    """ Load cluster data indexed by cluster. """

    @cached_property
    def data(self) -> pd.Series:
        data = pd.read_csv(self.path,
                           index_col=self.index_cols()).squeeze(axis=1)
        if not isinstance(data, pd.Series):
            raise ValueError(f"{self} must have one column, but got\n{data}")
        # Any numeric data in the header will be read as strings and
        # must be cast to integers using parse_header.
        header = parse_header(data.index)
        # The index must be replaced with the header index for the
        # type casting to take effect.
        data.index = header.index
        return data


class DeconvolveTabulator(PartialTabulator, ABC):

    @classmethod
    def table_types(cls):
        return [DeconvolvePositionTableWriter, DeconvolveAbundanceTableWriter]

    def __init__(self, *, num_clusters: int, **kwargs):
        self.deconv_count_mut_conf = kwargs.pop("deconv_count_mut_conf", False)
        self.cluster_list = kwargs.pop("cluster_list", None)
        self.positions_list = kwargs.pop("positions_list", None)
        self.confidences_list = kwargs.pop("confidences_list", None)

        super().__init__(**kwargs)
        if num_clusters is None:
            raise ValueError(
                f"{type(self).__name__} requires clusters, but got num_clusters={num_clusters}"
            )
        self.num_clusters = num_clusters
        self.ks = [self.num_clusters]

    @cached_property
    def no_probe_table(self):
        return (MaskPositionTableLoader(self.dataset.no_probe_path) 
                if self.dataset.no_probe_path else None)

    @cached_property
    def only_probe_table(self):
        return (MaskPositionTableLoader(self.dataset.only_probe_path) 
                if self.dataset.only_probe_path else None)

    @cached_property 
    def bayes(self):
        if self.no_probe_table and self.only_probe_table:
            return calc_bayes(self.no_probe_table,
                              self.only_probe_table,
                              self.dataset.deconv_pattern)

    @cached_property
    def _adjusted(self):
        table_per_pos = super(PartialTabulator, self).data_per_pos
        if self.min_mut_gap > 0:
            if self.region.length > np.sqrt(1_000_000_000):
                logger.warning("Using bias correction on a region with "
                               f"{self.region.length} positions requires "
                               ">1 GB of memory. If this is impractical, you "
                               "can (at the cost of lower accuracy) disable "
                               "bias correction using --min-mut-gap 0.")
            if self.deconv_count_mut_conf:
                orig_df = pd.DataFrame(index=table_per_pos.index, columns=table_per_pos.columns)
                for clust, pos, conf in zip(self.cluster_list, self.positions_list, self.confidences_list):
                    for pos_i, conf_i in zip(pos, conf):
                        for rel in (INFOR_REL, MUTAT_REL):
                            idx = ((pos_i, slice(None)), (rel, *clust))
                            orig_df.loc[idx] = table_per_pos.loc[idx].values[0]
                            if rel == INFOR_REL:
                                table_per_pos.loc[idx] = 1
                            elif rel == MUTAT_REL:
                                table_per_pos.loc[idx] = conf_i
            try:
                n_rels, n_clust = adjust_counts(table_per_pos,
                                     self.p_ends_given_clust_noclose,
                                     self.num_reads,
                                     self.region,
                                     self.min_mut_gap,
                                     self.quick_unbias,
                                     self.quick_unbias_thresh)
                if self.deconv_count_mut_conf:
                    for clust, pos in zip(self.cluster_list, self.positions_list):
                        for pos_i in pos:
                            for rel in (INFOR_REL, MUTAT_REL):
                                idx = ((pos_i, slice(None)), (rel, *clust))
                                n_rels.loc[idx] = orig_df.loc[idx].values[0]
                return n_rels, n_clust
            except Exception as error:
                logger.warning(error)
            
            if self.deconv_count_mut_conf:
                for clust, pos in zip(self.cluster_list, self.positions_list):
                    for pos_i in pos:
                        for rel in (INFOR_REL, MUTAT_REL):
                            idx = ((pos_i, slice(None)), (rel, *clust))
                            table_per_pos.loc[idx] = orig_df.loc[idx].values[0]
                    
        return table_per_pos, self.num_reads

    @cached_property
    def clust_header(self):
        """ Header of the per-cluster data. """
        return make_header(ks=[self.num_clusters])

    @cached_property
    def data_per_clust(self):
        """ Number of reads in each cluster. """
        n_rels, n_clust = self._adjusted
        n_clust.name = "Number of Reads"
        return n_clust


class DeconvolveDatasetTabulator(DeconvolveTabulator, PartialDatasetTabulator):

    @classmethod
    def load_function(cls):
        return load_deconvolve_dataset

    def __init__(self, *, dataset: DeconvolveMutsDataset, **kwargs):
        self.dataset = dataset
        super().__init__(dataset=dataset,
                         num_clusters=dataset.num_clusters,
                         **kwargs)
    
    def _get_named_index(self, index):
        return pd.MultiIndex.from_tuples([(relationship, self.dataset.cluster_index_to_name.get((k, cluster), f"Unmapped_{k}_{cluster}")) for relationship, k, cluster in index], names=["Relationship", "Name"])
