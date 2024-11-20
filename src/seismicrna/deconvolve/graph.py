from os.path import exists
from ..core.logs import logger
from seismicrna.core.header import K_CLUST_KEY
from seismicrna.graph.profile import OneRelProfileGraph
from seismicrna.graph.corroll import RollingCorrelationGraph
from seismicrna.deconvolve.table import DeconvolvePositionTable
from seismicrna.deconvolve.data import DeconvolveMutsDataset
import pandas as pd
from typing import Iterable
from collections import Counter
from functools import cached_property
import numpy as np

def format_clust_name(k: int, 
                      clust: int, 
                      mapping: dict,
                      clusters: Iterable[tuple[int, int]],
                      allow_zero: bool = False):
    index = np.where(clusters == (k, clust))[0][0]
    return list(mapping.keys())[index]
 
def format_clust_names(clusts: Iterable[tuple[int, int]],
                       mapping: dict,
                       index: Iterable[tuple[int, int]],
                       allow_zero: bool = False,
                       allow_duplicates: bool = False):
    """ Format pairs of k and cluster numbers into a list of names.

    Parameters
    ----------
    clusts: Iterable[tuple[int, int]]
        Zero or more pairs of k and cluster numbers.
    allow_zero: bool = False
        Allow k and cluster to be 0.
    allow_duplicates: bool = False
        Allow k and cluster pairs to be duplicated.

    Returns
    -------
    list[str]
        List of names of the pairs of k and cluster numbers.

    Raises
    ------
    ValueError
        If `allow_duplicates` is False and an k-cluster pair occurs
        more than once.
    """
    if not allow_duplicates:
        counts = Counter(clusts := list(clusts))
        if dups := [clust for clust, count in counts.items() if count > 1]:
            raise ValueError(f"Duplicate clusters: {dups}")
    return [format_clust_name(k, clust, mapping, index, allow_zero=allow_zero)
            for k, clust in clusts]

def _index_titles(index: pd.Index | None,
                  mapping: dict,
                  clusters: Iterable[tuple[int, int]]):
        return (format_clust_names(index, mapping, clusters, allow_zero=True, allow_duplicates=False)
                if index is not None
                else None)

def map_names(names, name_to_cluster_index):
    valid_names = [name for name in names if name in name_to_cluster_index]
    missing_names = set(names) - set(valid_names)
    for name in missing_names:
        logger.warning(f"Could not find cluster with name {name}")
    valid_indexes = [name_to_cluster_index[name] for name in valid_names]
    return valid_indexes

class DeconvolvedGraph:
    def __init__(self, **kwargs):
        if hasattr(self, "table"):
            return
        if (table := kwargs.get("table")) or (table := kwargs.get("table1")):
            if not hasattr(table, "tabulator"):
                if exists(table.report_path):
                    dataset = DeconvolveMutsDataset(table.report_path)
                else:
                    logger.error(f"No deconvolve report found for {table.ref} {table.sect}.")
            else:
                dataset = table.tabulator.dataset
            self.cluster_index_to_name = dataset.cluster_index_to_name
            self.name_to_cluster_index = dataset.name_to_cluster_index
            self.deconv_confs = dataset.deconv_confs
            self.read_counts = dataset.read_counts
            self.table = table
        else:
            logger.error("DeconvolvedGraph missing keyword argument table")
            


class DeconvolvedRollingCorrelationGraph(RollingCorrelationGraph):
    def __init__(self, **kwargs):
        DeconvolvedGraph.__init__(self, **kwargs)
        cluster_names = kwargs.pop("cluster_names", None)
        if cluster_names:
            if len(cluster_names) != 2:
                logger.error(f"cluster_names must be of length 2, but got {cluster_names}")
            k_clust1, k_clust2 = map_names(cluster_names, self.name_to_cluster_index)
            self.k1, self.clust1 = k_clust1
            self.k2, self.clust2 = k_clust2
        super().__init__(k1=self.k1,
                         clust1=self.clust1,
                         k2=self.k2,
                         clust2=self.clust2,
                         **kwargs)

        
    @cached_property
    def data1(self):
        """ Data from table 1. """
        return self._fetch_data(self.table1,
                                k=self.k1,
                                clust=self.clust1,
                                exclude_masked=True)
    @cached_property
    def data2(self):
        """ Data from table 2. """
        return self._fetch_data(self.table2,
                                k=self.k2,
                                clust=self.clust2,
                                exclude_masked=True)
    
    @cached_property
    def row_titles(self):
        """ Titles of the rows. """
        name_list = list(map(self.cluster_index_to_name.get, self.row_tracks))
        read_counts = [int(self.read_counts.get(name, 0.0)) for name in name_list]
        conf_list = list(map(self.deconv_confs.get, name_list))
        labels = list()
        for name, reads, conf in zip(name_list, read_counts, conf_list):
            if conf:
                labels.append(f"{name} ({conf*100:.2f}% confident)<br><sub>{reads:,} reads</sub>")
            else:
                labels.append(f"{name}<br><sub>{reads:,} reads</sub>")
        return labels

    @cached_property
    def col_titles(self):
        """ Titles of the columns. """
        name_list = list(map(self.cluster_index_to_name.get, self.col_tracks))
        read_counts = [int(self.read_counts.get(name, 0.0)) for name in name_list]
        conf_list = list(map(self.deconv_confs.get, name_list))
        labels = list()
        for name, reads, conf in zip(name_list, read_counts, conf_list):
            if conf:
                labels.append(f"{name} ({conf*100:.2f}% confident)<br><sub>{reads:,} reads</sub>")
            else:
                labels.append(f"{name}<br><sub>{reads:,} reads</sub>")
        return labels
    

class OneRelDeconvolvedProfileGraph(OneRelProfileGraph):
    """ Bar graph with one relationship per position. """
    def __init__(self, **kwargs):
        DeconvolvedGraph.__init__(self, **kwargs)
        names = kwargs.pop("names", None)
        if names:
            mapped_indexes = map_names(names, self.name_to_cluster_index)
            if mapped_indexes:
                kwargs[K_CLUST_KEY] = mapped_indexes
        super().__init__(**kwargs)

    @cached_property
    def row_titles(self):
        """ Titles of the rows. """
        name_list = list(map(self.cluster_index_to_name.get, self.row_tracks))
        read_counts = [int(self.read_counts.get(name, 0.0)) for name in name_list]
        conf_list = list(map(self.deconv_confs.get, name_list))
        labels = list()
        for name, reads, conf in zip(name_list, read_counts, conf_list):
            if conf:
                labels.append(f"{name}<br><sub>{conf*100:.2f}% confident<br>{reads:,} reads</sub>")
            else:
                labels.append(f"{name}<br><sub>{reads:,} reads</sub>")
        return labels
        