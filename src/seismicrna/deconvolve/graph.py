#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:48:06 2024

@author: justin
"""

from seismicrna.graph.profile import OneRelProfileGraph
from seismicrna.graph.corroll import RollingCorrelationGraph
import pandas as pd
from typing import Iterable
from collections import Counter
from functools import cached_property
import numpy as np

def format_clust_name(order: int, 
                      clust: int, 
                      mapping: dict,
                      clusters: Iterable[tuple[int, int]],
                      allow_zero: bool = False):
    index = np.where(clusters == (order, clust))[0][0]
    return list(mapping.keys())[index]
 
def format_clust_names(clusts: Iterable[tuple[int, int]],
                       mapping: dict,
                       index: Iterable[tuple[int, int]],
                       allow_zero: bool = False,
                       allow_duplicates: bool = False):
    """ Format pairs of order and cluster numbers into a list of names.

    Parameters
    ----------
    clusts: Iterable[tuple[int, int]]
        Zero or more pairs of order and cluster numbers.
    allow_zero: bool = False
        Allow order and cluster to be 0.
    allow_duplicates: bool = False
        Allow order and cluster pairs to be duplicated.

    Returns
    -------
    list[str]
        List of names of the pairs of order and cluster numbers.

    Raises
    ------
    ValueError
        If `allow_duplicates` is False and an order-cluster pair occurs
        more than once.
    """
    if not allow_duplicates:
        counts = Counter(clusts := list(clusts))
        if dups := [clust for clust, count in counts.items() if count > 1]:
            raise ValueError(f"Duplicate clusters: {dups}")
    return [format_clust_name(order, clust, mapping, index, allow_zero=allow_zero)
            for order, clust in clusts]

def _index_titles(index: pd.Index | None,
                  mapping: dict,
                  clusters: Iterable[tuple[int, int]]):
        print("Using new index")
        return (format_clust_names(index, mapping, clusters, allow_zero=True, allow_duplicates=False)
                if index is not None
                else None)

class DeconvolvedRollingCorrelationGraph(RollingCorrelationGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @cached_property
    def data1(self):
        """ Data from table 1. """
        return self._fetch_data(self.table1,
                                order=self.order1,
                                clust=self.clust1,
                                exclude_masked=True)
    @cached_property
    def data2(self):
        """ Data from table 2. """
        return self._fetch_data(self.table2,
                                order=self.order2,
                                clust=self.clust2,
                                exclude_masked=True)
    

class OneRelDeconvolvedProfileGraph(OneRelProfileGraph):
    """ Bar graph with one relationship per position. """
    def __init__(self, *, mapping: dict, clusters: Iterable[tuple[int, int]], **kwargs):
        self.mapping = mapping
        self.clusters = clusters
        super().__init__(**kwargs)
        
    @cached_property
    def data(self):
        return self._fetch_data(self.table,
                                order=self.order,
                                clust=self.clust,
                                order_clust_list=self.order_clust_list,
                                exclude_masked = True)
        
        
    @cached_property
    def row_titles(self):
        """ Titles of the rows. """
        return _index_titles(self.row_index, self.mapping, self.clusters)
        