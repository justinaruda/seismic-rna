from abc import ABC
from functools import cached_property

import pandas as pd
import numpy as np

from .data import DeconvolveMutsDataset
from ..core import path
from ..core.header import (ClustHeader,
                           RelClustHeader,
                           make_header,
                           parse_header)
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
from ..core.table.base import (MATCH_REL,
                               MUTAT_REL,
                               INFOR_REL)
from ..core.logs import logger
from ..core.write import need_write
from ..mask.table import adjust_counts


class DeconvolveTable(RelTypeTable, ABC):

    @classmethod
    def kind(cls):
        return path.CMD_DECONV_DIR

    @classmethod
    def header_type(cls):
        return RelClustHeader


class DeconvolvePosTable(DeconvolveTable, PartialPositionTable, ABC):
    pass


class DeconvolveAbundanceTable(AbundanceTable, PartialTable, ABC):

    @classmethod
    def kind(cls):
        return path.CMD_DECONV_DIR

    @classmethod
    def header_type(cls):
        return ClustHeader

    @classmethod
    def index_depth(cls):
        return cls.header_depth()

    def _get_header(self):
        return parse_header(self.data.index)


class DeconvolvePosTableWriter(PositionTableWriter, DeconvolvePosTable):
    
    def write(self, force: bool):
        """ Write the table's rounded data to the table's CSV file. """
        if need_write(self.path, force):
            orig_cols = self.data.columns
            self.data.columns = self.tabulator._get_named_index(
                                                            self.data.columns)
            self.data.round(decimals=PRECISION).to_csv(self.path)
            self.data.columns = orig_cols
            logger.routine(f"Wrote {self} to {self.path}")
        return self.path


class DeconvolveAbundanceTableWriter(AbundanceTableWriter,
                                     DeconvolveAbundanceTable):
    pass


class DeconvolvePosTableLoader(PositionTableLoader, DeconvolvePosTable):
    """ Load cluster data indexed by position. """
    
    @cached_property
    def data(self) -> pd.DataFrame:
        data = pd.read_csv(self.path,
                           index_col=self.index_cols(),
                           header=[0,1])
        # Any numeric data in the header will be read as strings and
        # must be cast to integers using parse_header.
        header = parse_header(data.columns)
        # The columns must be replaced with the header index for the
        # type casting to take effect.
        data.columns = header.index
        return data


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
        return [DeconvolvePosTableWriter, DeconvolveAbundanceTableWriter]

    def __init__(self, *, num_clusters: int, **kwargs):
        self.corr_editing_bias = kwargs.pop("corr_editing_bias", False)
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
    def _adjusted(self):
        table_per_pos = super(PartialTabulator, self).data_per_pos
        if self.min_mut_gap > 0:
            if self.section.length > np.sqrt(1_000_000_000):
                logger.warning("Using bias correction on a section with "
                               f"{self.section.length} positions requires "
                               ">1 GB of memory. If this is impractical, you "
                               "can (at the cost of lower accuracy) disable "
                               "bias correction using --min-mut-gap 0.")
            if self.corr_editing_bias:
                orig_df = pd.DataFrame(index=table_per_pos.index, columns=table_per_pos.columns)
                for clust, pos, conf in zip(self.cluster_list, self.positions_list, self.confidences_list):
                    for pos_i, conf_i in zip(pos, conf):
    
                        idx_infor = ((pos_i, slice(None)), (INFOR_REL, *clust))
                        idx_mutat = ((pos_i, slice(None)), (MUTAT_REL, *clust))
    
                        orig_df.loc[idx_infor] = table_per_pos.loc[idx_infor].values[0]
                        orig_df.loc[idx_mutat] = table_per_pos.loc[idx_mutat].values[0]
                
                        table_per_pos.loc[idx_infor] = 1
                        table_per_pos.loc[idx_mutat] = conf_i
            try:
                n_rels, n_clust = adjust_counts(table_per_pos,
                                     self.p_ends_given_clust_noclose,
                                     self.num_reads,
                                     self.section,
                                     self.min_mut_gap,
                                     self.quick_unbias,
                                     self.quick_unbias_thresh)
                if self.corr_editing_bias:
                    for clust, pos in zip(self.cluster_list, self.positions_list):
                        for pos_i in pos:
                            idx_infor = ((pos_i, slice(None)), (INFOR_REL, *clust))
                            idx_mutat = ((pos_i, slice(None)), (MUTAT_REL, *clust))
                            n_rels.loc[idx_infor] = orig_df.loc[idx_infor].values[0]
                            n_rels.loc[idx_mutat] = orig_df.loc[idx_mutat].values[0]
                return n_rels, n_clust
            except Exception as error:
                logger.warning(error)
            
            if self.corr_editing_bias:
                for clust, pos in zip(self.cluster_list, self.positions_list):
                    for pos_i in pos:
                        idx_infor = ((pos_i, slice(None)), (INFOR_REL, *clust))
                        idx_mutat = ((pos_i, slice(None)), (MUTAT_REL, *clust))
                        table_per_pos.loc[idx_infor] = orig_df.loc[idx_infor].values[0]
                        table_per_pos.loc[idx_mutat] = orig_df.loc[idx_mutat].values[0]
                    
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

    def __init__(self, *, dataset: DeconvolveMutsDataset, **kwargs):
        self.dataset = dataset
        super().__init__(dataset=dataset,
                         num_clusters=dataset.num_clusters,
                         **kwargs)
    
    def _get_named_index(self, index):
        return pd.MultiIndex.from_tuples([(relationship, self.dataset.cluster_index_to_name.get((k, cluster), f"Unmapped_{k}_{cluster}")) for relationship, k, cluster in index], names=["Relationship", "Name"])
