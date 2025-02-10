from functools import cached_property

import numpy as np
import pandas as pd

from ..mask.batch import PartialRegionMutsBatch, PartialReadBatch


class ClusterReadBatch(PartialReadBatch):

    def __init__(self, *, resps: pd.DataFrame, **kwargs):
        self.resps = resps
        super().__init__(**kwargs)

    @cached_property
    def num_reads(self) -> pd.Series:
        return self.resps.sum(axis=0)

    @cached_property
    def read_nums(self):
        return self.resps.index.values


class ClusterMutsBatch(ClusterReadBatch, PartialRegionMutsBatch):

    @property
    def read_weights(self):
        read_weights = self.resps
        if self.masked_reads_bool.any():
            read_weights[self.masked_reads_bool] = 0
        return read_weights
