import os
from functools import cached_property

import numpy as np
import pandas as pd
from click import command

from .base import get_action_name, make_tracks
from .color import ColorMapGraph, RelColorMap
from .dataset import DatasetGraph, DatasetGraphWriter, DatasetGraphRunner
from .hist import COUNT_NAME
from .trace import get_hist_trace
from ..core.arg import opt_mutdist_null
from ..core.header import REL_NAME, make_header
from ..core.seq import FIELD_END5, FIELD_END3
from ..core.table import PositionTable, all_patterns
from ..core.unbias import (calc_p_noclose_given_ends_auto,
                           calc_p_ends_observed,
                           triu_dot)
from ..table import get_tabulator_type

COMMAND = __name__.split(os.path.extsep)[-1]
NULL_SUFFIX = "-NULL"


def get_null_name(name: str):
    return f"{name}{NULL_SUFFIX}"


class MutationDistanceGraph(DatasetGraph, ColorMapGraph):
    """ Distance between the closest two mutations in each read. """

    @classmethod
    def graph_kind(cls):
        return COMMAND

    @classmethod
    def what(cls):
        return "Length of smallest distance"

    @classmethod
    def get_cmap_type(cls):
        return RelColorMap

    def __init__(self, *, mutdist_null: bool, **kwargs):
        super().__init__(**kwargs)
        self.calc_null = mutdist_null

    @cached_property
    def action(self):
        return get_action_name(self.dataset)

    @property
    def x_title(self):
        return "Smallest distance between two mutations in a read (nt)"

    @property
    def y_title(self):
        return "Number of reads"

    @cached_property
    def row_tracks(self):
        return make_tracks(self.dataset, self.k, self.clust)

    @cached_property
    def _data(self):
        num_bins = self.dataset.region.length
        header = make_header(rels=self.rel_names, ks=self.dataset.ks)
        zero = 0. if header.clustered() else 0
        hists = pd.DataFrame(zero,
                             pd.RangeIndex(num_bins, name=COUNT_NAME),
                             header.index)
        batch_counts = list()
        max_read_length = 0
        # For each read in all batches, calculate the minimum distance
        # between two mutations.
        for batch in self.dataset.iter_batches():
            batch_counts.append(batch.count_all(
                patterns=all_patterns(self.dataset.pattern),
                ks=self.dataset.ks,
                count_ends=True,
                count_pos=True,
                count_read=False
            ))
            max_read_length = max(max_read_length,
                                  batch.read_lengths.max(initial=0))
            min_mut_dist = batch.calc_min_mut_dist(self.pattern)
            if header.clustered():
                if not isinstance(batch.read_weights, pd.DataFrame):
                    raise TypeError("batch.read_weights must be DataFrame, "
                                    f"but got {batch.read_weights}")
                for (k, clust), weights in batch.read_weights.items():
                    col = (self.rel_name, k, clust)
                    hists.loc[:, col] += np.bincount(min_mut_dist,
                                                     weights=weights,
                                                     minlength=num_bins)
            else:
                if batch.read_weights is not None:
                    raise TypeError("batch.read_weights must be None, "
                                    f"but got {batch.read_weights}")
                hists.loc[:, self.rel_name] += np.bincount(min_mut_dist,
                                                           minlength=num_bins)
        # Calculate the mutation rates and 5'/3' ends.
        dataset_type = type(self.dataset)
        tabulator_type = get_tabulator_type(dataset_type, count=True)
        init_keywords = (set(get_tabulator_type(dataset_type).init_kws())
                         - {"get_batch_count_all", "num_batches"})
        kwargs = {kw: getattr(self.dataset, kw) for kw in init_keywords}
        tabulator = tabulator_type(batch_counts=batch_counts,
                                   count_ends=True,
                                   count_pos=True,
                                   count_read=False,
                                   **kwargs)
        return hists, tabulator, max_read_length

    @property
    def hists(self):
        hists, tabulator, max_read_length = self._data
        return hists

    @property
    def tabulator(self):
        hists, tabulator, max_read_length = self._data
        return tabulator

    @property
    def max_read_length(self):
        hists, tabulator, max_read_length = self._data
        return max_read_length

    @cached_property
    def table(self):
        tables = list(self.tabulator.generate_tables(pos=True,
                                                     read=False,
                                                     clust=False))
        assert len(tables) == 1
        table = tables[0]
        assert isinstance(table, PositionTable)
        return table

    @cached_property
    def _real_hist(self):
        if self.table.header.clustered():
            cols = self.rel_name, slice(self.k), slice(self.clust)
            return self.hists.loc[:, cols]
        return self.hists

    @cached_property
    def _null_hist(self):
        if self.table.header.clustered():
            clusters = slice(self.k), slice(self.clust)
            end_counts = self.tabulator.end_counts.loc[:, clusters]
            num_reads = self.tabulator.num_reads.loc[clusters].values
        else:
            end_counts = self.tabulator.end_counts
            num_reads = self.tabulator.num_reads
        end5 = self.dataset.region.end5
        end5s = np.asarray(end_counts.index.get_level_values(FIELD_END5) - end5)
        end3s = np.asarray(end_counts.index.get_level_values(FIELD_END3) - end5)
        p_ends = np.atleast_3d(calc_p_ends_observed(self.dataset.region.length,
                                                    end5s,
                                                    end3s,
                                                    end_counts.values))
        p_mut = self.table.fetch_ratio(k=self.k, rel=self.rel_name)
        # For every possible gap, calculate the fraction of reads that
        # have no two mutations closer than that gap.
        p_noclose_gap = np.empty((self.max_read_length,
                                  self._real_hist.columns.size),
                                 dtype=float)
        p_noclose_gap[0] = 1.
        for gap in range(1, self.max_read_length):
            p_noclose_ends = calc_p_noclose_given_ends_auto(p_mut.values, gap)
            p_noclose_gap[gap] = triu_dot(p_noclose_ends, p_ends)
        # For every possible distance, calculate the fraction of reads
        # where the closest two mutations have exactly that distance,
        # and for 0 the fraction of reads with fewer than two mutations.
        p_dist = np.zeros_like(self._real_hist, dtype=float)
        p_dist[0] = p_noclose_gap[self.max_read_length - 1]
        p_dist[1: self.max_read_length] = -np.diff(p_noclose_gap, axis=0)
        assert np.all(p_dist >= 0.)
        assert np.allclose(p_dist.sum(axis=0), 1.)
        # Multiply by the number of reads to obtain the histogram.
        return pd.DataFrame(
            p_dist * num_reads,
            index=self._real_hist.index,
            columns=self._real_hist.columns
        ).rename(columns=get_null_name, level=REL_NAME, copy=False)

    @cached_property
    def g_test(self):
        """ G-test statistic and P-value. """
        if self.calc_null:
            observed = self._real_hist.values
            expected = self._null_hist.values
            assert observed.ndim == 2
            assert observed.shape == expected.shape
            assert np.allclose(observed.sum(axis=0), expected.sum(axis=0))
            n, k = observed.shape
            dof = n - 1
            if dof >= 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    g_stat = 2. * np.where(
                        np.logical_and(observed > 0, expected > 0),
                        observed * np.log(observed / expected),
                        0.
                    ).sum(axis=0)
                from scipy.stats import chi2
                p_value = 1. - chi2.cdf(g_stat, dof)
            else:
                g_stat = np.zeros(k, dtype=float)
                p_value = np.ones_like(g_stat)
        else:
            g_stat = np.nan
            p_value = np.nan
        return (pd.Series(g_stat, index=self._real_hist.columns),
                pd.Series(p_value, index=self._real_hist.columns))

    @cached_property
    def data(self):
        if self.calc_null:
            return pd.concat([self._real_hist, self._null_hist], axis=1)
        return self._real_hist

    def get_traces(self):
        if self.calc_null:
            for row, ((_, real_values), (_, null_values)) in enumerate(
                    zip(self._real_hist.items(),
                        self._null_hist.items(),
                        strict=True),
                    start=1
            ):
                yield (row, 1), get_hist_trace(real_values,
                                               self.rel_name,
                                               self.cmap)
                yield (row, 1), get_hist_trace(null_values,
                                               get_null_name(self.rel_name),
                                               self.cmap)
        else:
            for row, (_, real_values) in enumerate(
                    self._real_hist.items(),
                    start=1
            ):
                yield (row, 1), get_hist_trace(real_values,
                                               self.rel_name,
                                               self.cmap)


class MutationDistanceWriter(DatasetGraphWriter):

    def get_graph(self, rel, **kwargs):
        return MutationDistanceGraph(dataset=self.dataset,
                                     rel=rel,
                                     **kwargs)


class MutationDistanceRunner(DatasetGraphRunner):

    @classmethod
    def get_writer_type(cls):
        return MutationDistanceWriter

    @classmethod
    def var_params(cls):
        return super().var_params() + [opt_mutdist_null]


@command(COMMAND, params=MutationDistanceRunner.params())
def cli(*args, **kwargs):
    """ Distance between the closest two mutations in each read. """
    return MutationDistanceRunner.run(*args, **kwargs)

########################################################################
#                                                                      #
# © Copyright 2022-2025, the Rouskin Lab.                              #
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
