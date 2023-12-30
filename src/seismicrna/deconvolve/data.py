from functools import cached_property, cache

from ..clust.batch import ClustMutsBatch
from ..clust.io import ClustBatchIO
from ..core.data import (BatchedLoadedDataset,
                         BatchedMergedDataset,
                         MergedMutsDataset)
from ..core.header import index_orders_clusts
from ..mask.batch import MaskMutsBatch
from ..mask.report import MaskReport
from ..relate.data import RelateLoader
from ..relate.batch import RelateRefseqBatch
from ..mask.data import MaskLoader

from ..mask.batch import apply_mask
from ..mask.io import MaskBatchIO
from ..mask.data import MaskMerger, MaskLoader
from ..clust.data import ClustLoader
from ..core.rel.pattern import RelPattern

from ..table.calc import MaskTabulator, ClustTabulator
from ..table.write import MaskPosTableWriter, ClustPosTableWriter

from ..graph.seqbar import ClusterSingleRelSeqBarGraph

import numpy as np
import pandas as pd

class Deconvolver:
    def __init__(self, *, deconvolve_report, edited_report, background_report, positions, mut_val, pattern, strict):
        self.deconvolve_dataset = MaskMerger.load(deconvolve_report)
        self.edited_dataset = MaskMerger.load(edited_report)
        self.background_dataset = MaskMerger.load(background_report)
        self.positions = positions
        self.mut_val = mut_val
        self.pattern = pattern
        self.strict = strict
    
    def _get_deconvolve_merger(self):
        self.deconvolve_merger = DeconvolveMerger(self.positions, self.mut_val, self.pattern, self.deconvolve_dataset, None, strict=self.strict)
    
    @cached_property
    def edited_tabulator(self):
        return MaskPosTableWriter(MaskTabulator(self.edited_dataset))
    
    @cached_property
    def background_tabulator(self):
        return MaskPosTableWriter(MaskTabulator(self.background_dataset))
    
    @cached_property
    def bayes(self):
        self.edited_mus = next(self.edited_tabulator.iter_profiles()).data
        self.background_mus = next(self.background_tabulator.iter_profiles()).data
        bayes = 0.98 * (self.edited_mus)/(self.background_mus + self.edited_mus)
        return bayes[bayes.index.get_level_values(1).isin(["A"])]

    def set_positions(self, new_positions):
        self.positions = new_positions

    @cached_property
    def cluster_tabulator(self):
        return ClustTabulator(self.deconvolve_merger)
    
    @cached_property
    def table_writer(self):
        return ClustPosTableWriter(self.cluster_tabulator)

    def graph(self, *, force=True):
        graph_obj = ClusterSingleRelSeqBarGraph(order=2, table=self.table_writer, rels="m", y_ratio=True, quantile=0)
        graph_obj.write_html(force=force)
        
    def deconvolve(self, positions, mut_val, pattern):
        self.positions = positions
        self.mut_val = mut_val
        self.pattern = pattern
        self._get_deconvolve_merger()
        self.graph(force=True)
        

class DeconvolveMerger(BatchedMergedDataset, MergedMutsDataset):
    """ Merge deconvolved responsibilities with mutation data. """

    def __init__(self, positions: np.ndarray[int], mut_val: int, pattern: RelPattern, data1, data2, strict=False):
        self.positions = positions
        self.mut_val = mut_val
        self.strict = strict
        self.max_order = 2
        self.accum_pattern = pattern
        super().__init__(data1, data2)

    @classmethod
    def get_data_type(cls):
        return ClustMutsBatch

    @classmethod
    def get_dataset1_type(cls):
        return MaskMerger

    @classmethod
    def get_dataset2_type(cls):
        return type(None)

    @property
    def min_mut_gap(self):
        return self.data1.min_mut_gap

    @property
    def pattern(self):
        return self.accum_pattern

    @cached_property
    def section(self):
        return self.data1.section

    @cached_property
    def max_order(self):
        return self.max_order
    
    @cached_property
    def clusters(self):
        return index_orders_clusts(self.max_order)
    
    def _get_data_attr(self, name: str):
        val1 = getattr(self.data1, name)
        return val1
    
    def _get_edited(self, batch):
        positions = self.positions
        mut_val = self.mut_val
        strict = self.strict
        muts = batch.muts
        edited = None
        unedited = None
        for pos in positions:
            if edited is None:
                edited = muts.get(pos, dict()).get(mut_val, np.array([]))
            else:
                # Pick intersection or union.
                edited = np.intersect1d(edited, muts.get(pos, dict()).get(mut_val, np.array([])))
        unedited = np.setdiff1d(batch.read_nums, edited)
        if strict:
            remove = np.array([])
            for pos in muts:
                if pos in positions:
                    continue
                muts_at_pos = muts.get(pos, dict()).get(mut_val, np.array([]))
                not_mut_at_pos = np.setdiff1d(batch.read_nums, muts_at_pos)
                edited_at_pos = np.setdiff1d(edited, not_mut_at_pos)
                remove = np.union1d(remove, edited_at_pos)
            edited = np.setdiff1d(edited, remove)
            unedited = np.setdiff1d(unedited, remove)
        self.edited = edited
        self.unedited = unedited
    
    def _build_resps(self, batch: MaskMutsBatch):
        resps_matrix = np.zeros((batch.num_reads, 3), dtype=int)
        resps_matrix[batch.read_indexes, 0] = 1
        resps_matrix[batch.read_indexes[self.edited], 1] = 1
        resps_matrix[batch.read_indexes[self.unedited], 2] = 1
        resps = pd.DataFrame(resps_matrix, index=batch.read_nums, columns=self.clusters)
        return resps
        
    def _iter_batches(self):
        for batch1 in self.data1.iter_batches():
            self._get_edited(batch1)
            resps = self._build_resps(batch1)
            # masked_batch = apply_mask(batch1, positions=np.setdiff1d(batch1.pos_nums, self.positions))
            yield self._merge(batch1, resps)

    def _merge(self, batch1: MaskMutsBatch, resps: pd.DataFrame):
        return self.get_data_type()(batch=batch1.batch,
                                    refseq=batch1.refseq,
                                    muts=batch1.muts,
                                    end5s=batch1.end5s,
                                    mid5s=batch1.mid5s,
                                    mid3s=batch1.mid3s,
                                    end3s=batch1.end3s,
                                    resps=resps,
                                    sanitize=False)