from functools import cached_property, cache

from ..cluster.batch import ClusterMutsBatch
from ..cluster.data import ClusterMutsDataset
from ..core.header import index_orders_clusts
from ..mask.batch import MaskMutsBatch
from ..mask.data import MaskReadDataset, MaskMutsDataset
from ..core.rel.pattern import RelPattern

from ..table.calc import MaskTabulator, ClustTabulator
from ..table.write import MaskPosTableWriter, ClustPosTableWriter

from ..graph.profile import OneRelProfileGraph
from ..graph.corroll import RollingCorrelationGraph

from .report import DeconvolveReport

from ..cluster.uniq import UniqReads, get_uniq_reads

from .graph import OneRelDeconvolvedProfileGraph, DeconvolvedRollingCorrelationGraph
# from ..graph.corroll import RollingCorrelationGraph

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import math

class Deconvolver:
    def __init__(self, 
                 *, 
                 deconvolve_report, 
                 edited_report, 
                 background_report, 
                 positions, 
                 mut_val, 
                 pattern, 
                 min_reads=1000, 
                 strict=False,
                 norm_edits=True):
        self.deconvolve_dataset = MaskMutsDataset.load(deconvolve_report)
        self.edited_dataset = MaskMutsDataset.load(edited_report)
        self.background_dataset = MaskMutsDataset.load(background_report)
        self.positions = positions
        self.mut_val = mut_val
        self.pattern = pattern
        self.min_reads = min_reads
        self.strict = strict
        self.sample = self.deconvolve_dataset.sample
        self.ref = self.deconvolve_dataset.ref
        self.sect = self.deconvolve_dataset.sect
        self.n_batches = 3
        self.section = self.deconvolve_dataset.section
        self.norm_edits = norm_edits

    def _get_deconvolve_merger(self):
        self.deconvolve_merger = DeconvolveMerger(self.positions, 
                                                  self.mut_val, 
                                                  self.pattern, 
                                                  self.deconvolve_dataset, 
                                                  None, 
                                                  min_reads=self.min_reads, 
                                                  strict=self.strict,
                                                  uniq_reads=self.uniq_reads,
                                                  norm_edits=self.norm_edits)

    @cached_property
    def edited_table(self):
        return MaskPosTableWriter(MaskTabulator(self.edited_dataset))

    @cached_property
    def background_table(self):
        return MaskPosTableWriter(MaskTabulator(self.background_dataset))

    @cached_property
    def bayes(self):
        self.edited_mus = next(self.edited_table.iter_profiles()).data
        self.background_mus = next(self.background_table.iter_profiles()).data
        bayes = 0.98 * (self.edited_mus)/(self.background_mus + self.edited_mus)
        return bayes[bayes.index.get_level_values(1).isin(["A"])]

    def set_positions(self, new_positions):
        self.positions = new_positions

    @cached_property
    def cluster_tabulator(self):
        return ClustTabulator(self.deconvolve_merger)

    @cached_property
    def deconvolve_tabulator(self):
        return MaskTabulator(self.deconvolve_dataset)

    @cached_property
    def table_writer(self):
        return ClustPosTableWriter(self.cluster_tabulator)

    @property
    def mapping_indexes(self):
        mapping_dict = {"pop_average": 0}
        cluster_index = 1
        for position in self.positions:
            pos_str_base = "_".join(map(str, position))
            pos_str_edited = f"edited_{pos_str_base}"
            pos_str_unedited = f"unedited_{pos_str_base}"
            mapping_dict[pos_str_edited] = cluster_index
            mapping_dict[pos_str_unedited] = cluster_index + 1
            cluster_index += 2
        return mapping_dict

    @property 
    def mapping_as_clusters(self):
        return {cluster_name: self.deconvolve_merger.clusters[cluster_num] 
                for cluster_name, cluster_num in self.mapping_indexes.items()}

    @property
    def order_cluster(self):
        index = list(self.mapping_as_clusters.keys())
        values = list(self.mapping_as_clusters.values())
        return pd.Series(values, index=index)

    def graph(self, order=None, clust=None, deconvolved_clusters=None, force=True):
        order_clust_list = list(self.order_cluster[deconvolved_clusters])
        mask_section = self.section.copy()
        mask_section.add_mask(mask_pos=self.positions.ravel(), name='mask-deconvolve')
        self.graph_obj = OneRelDeconvolvedProfileGraph(section=mask_section,
                                                       order=order, 
                                                       clust=clust, 
                                                       order_clust_list=order_clust_list,
                                                       mapping = self.mapping_as_clusters,
                                                       clusters = self.deconvolve_merger.clusters,
                                                       table=self.table_writer,
                                                       rel="m",
                                                       use_ratio=True,
                                                       quantile=0,
                                                       out_dir=Path("out"))
        self.graph_obj.write_html(force=force)

    def graph_roll_corr(self,
                        deconvolved_clust_1,
                        deconvolved_clust_2,
                        quantile=0., 
                        metric="pcc",
                        force=True):
        mask_section = self.section.copy()
        mask_section.mask_gu()
        mask_section.add_mask(mask_pos=self.positions.ravel(), name='mask-deconvolve')
        order1, clust1 = self.order_cluster[deconvolved_clust_1]
        order2, clust2 = self.order_cluster[deconvolved_clust_2]
        self.graph_obj = DeconvolvedRollingCorrelationGraph(section=mask_section,
                                                            metric=metric,
                                                            table1=self.table_writer, 
                                                            order1= order1, 
                                                            clust1=clust1, 
                                                            table2=self.table_writer, 
                                                            order2=order2, 
                                                            clust2=clust2, 
                                                            rel="m", 
                                                            window=45, 
                                                            winmin=15, 
                                                            use_ratio=True, 
                                                            quantile=quantile, 
                                                            out_dir=Path("out"))
        self.graph_obj.write_html(force=force)

    @property
    def read_counts(self):
        read_count_dict = dict()
        for cluster, read_count in enumerate(
                self.cluster_tabulator.table_per_clust.round(2)):
            name = next((name for name, mapped_cluster \
                         in self.mapping_indexes.items() \
                         if cluster == mapped_cluster), None)
            read_count_dict[name] = read_count
        return read_count_dict

    def deconvolve(self, positions=None, mut_val=None, pattern=None, force_write=True):
        # When deconvolving new position(s), clear cache to allow its recalculation.
        self.__dict__.pop("uniq_reads", None)
        self.__dict__.pop("cluster_tabulator", None)
        self.__dict__.pop("table_writer", None)
        self.positions = positions if positions is not None else self.positions
        self.mut_val = mut_val if mut_val is not None else self.mut_val
        self.pattern = pattern if pattern is not None else self.pattern
        self._get_deconvolve_merger()
        self.began = datetime.now()
        ClustPosTableWriter(self.cluster_tabulator).write(force=force_write)
        self.ended = datetime.now()
        self.report = DeconvolveReport.from_deconvolver(self)
        self.report.save(top=Path("out"), force=True)
    
    @cached_property
    def uniq_reads(self):
        a_pos = (np.array(list(self.section.seq)) != "A").nonzero()[0]
        self.section.add_mask(mask_pos=a_pos+self.section.coord[0], name="mask-non-a")
        self.section.add_mask(mask_pos=self.positions.ravel(), name="mask-deconvolved")
        only_ag = RelPattern.from_counts(discount=["ac", "at", "ca", "cg", "ct", "ta", "tg", "tc", "ga", "gc", "gt"], count_del=False, count_ins=False)
        uniq_reads = UniqReads(self.sample,
                               self.section,
                               0,
                               *get_uniq_reads(self.section.unmasked_int, 
                                               only_ag, 
                                               self.deconvolve_dataset.iter_batches()))
        self.section.remove_mask(name="mask-non-a")
        self.section.remove_mask(name="mask-deconvolved")
        return uniq_reads
        
        
class DeconvolveMerger(ClusterMutsDataset):
    """ Merge deconvolved responsibilities with mutation data. """

    def __init__(self, positions: np.ndarray[int], 
                 mut_val: int, 
                 pattern: RelPattern, 
                 data1, 
                 data2, 
                 min_reads=1000, 
                 strict=False,
                 uniq_reads=None,
                 norm_edits=True):
        self.positions = positions
        self.mut_val = mut_val
        self.strict = strict
        self.accum_pattern = pattern
        self.min_reads = min_reads
        self.uniq_reads = uniq_reads
        self.norm_edits = norm_edits

        super().__init__(data1, data2)

    @classmethod
    def get_data_type(cls):
        return ClusterMutsBatch

    @classmethod
    def get_dataset1_type(cls):
        return MaskReadDataset

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
        j = (len(self.positions) *2) + 1 # One extra column for population average
        # Solving the quadratic equation i^2 + i - 2j = 0
        # Using the quadratic formula i = (-b + sqrt(b^2 - 4ac)) / 2a, here a = 1, b = 1, c = -2j
        a, b, c = 1, 1, -2 * j
        discriminant = b**2 - 4 * a * c
        # Two solutions, but we're interested in the positive one
        i_positive = (-b + math.sqrt(discriminant)) / (2 * a)
        # Since i must be an integer, and we want the minimum i such that sum is >= j, we (up
        i_min = math.ceil(i_positive)
        return i_min
    
    @cached_property
    def clusters(self):
        return index_orders_clusts(self.max_order)
    
    def _get_data_attr(self, name: str):
        val1 = getattr(self.data1, name)
        return val1
    
    def _assign_uniq(self, batch, read_nums):
        return self.uniq_reads.batch_to_uniq[batch.batch][read_nums]
    
    def _get_edited(self, batch, positions, norm_edits=True):
        mut_val = self.mut_val
        strict = self.strict
        norm_edits = self.norm_edits
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

        uniq_ed, uniq_ed_counts = np.unique(self._assign_uniq(batch, edited), return_counts=True)
        uniq_uned, uniq_uned_counts = np.unique(self._assign_uniq(batch, unedited), return_counts=True)

        edited_indexes = np.isin(uniq_ed, uniq_uned)
        unedited_indexes = np.isin(uniq_uned, uniq_ed)
        min_intersect = np.minimum(uniq_ed_counts[edited_indexes],
                                   uniq_uned_counts[unedited_indexes])

        if norm_edits:
            keep_ed = np.zeros_like(edited, dtype=bool)
            keep_uned = np.zeros_like(unedited, dtype=bool)
            batch_to_uniq = self.uniq_reads.batch_to_uniq[batch.batch]
            for read_count, common_uniq_num in zip(min_intersect, uniq_ed[edited_indexes]):
                
                mask = batch_to_uniq == common_uniq_num
                read_nums = batch_to_uniq[mask].index.values
                
                ed_uniq = edited[np.isin(edited, read_nums)]
                uned_uniq = unedited[np.isin(unedited, read_nums)]
                
                keep_ed_nums = np.random.choice(ed_uniq, read_count, replace=False)
                keep_uned_nums = np.random.choice(uned_uniq, read_count, replace=False)
                
                keep_ed[np.isin(edited, keep_ed_nums)] = True
                keep_uned[np.isin(unedited, keep_uned_nums)] = True
            
            assert sum(keep_ed) == sum(keep_uned)
            edited = edited[keep_ed]
            unedited = unedited[keep_uned]

        
        self.edited = edited
        self.unedited = unedited

    def _populate_resps(self,
                        cluster: int, 
                        batch: MaskMutsBatch, 
                        resps: pd.DataFrame):
        if cluster == 1:
            resps.iloc[:,0] = np.array([1]*batch.num_reads)
        # if len(self.edited) < self.min_reads \
        #     or len(self.unedited) < self.min_reads:
        #     return resps
        resps_matrix = np.zeros((batch.num_reads, 2), dtype=int)
        resps_matrix[batch.read_indexes[self.edited], 0] = 1
        resps_matrix[batch.read_indexes[self.unedited], 1] = 1
        resps.loc[:,resps.columns[cluster]] = resps_matrix[:, 0]
        resps.loc[:,resps.columns[cluster+1]] = resps_matrix[:, 1]
        return resps

    def get_batch(self, batch_num: int):
        positions = self.positions
        batch1 = self.data1.get_batch(batch_num)
        resps = pd.DataFrame(index=batch1.read_nums, 
                             columns=self.clusters, 
                             dtype=int)
        cluster = 1
        for position in positions:
            self._get_edited(batch1, position)
            resps = self._populate_resps(cluster, batch1, resps)
            cluster += 2
            # masked_batch = apply_mask(batch1, positions=np.setdiff1d(batch1.pos_nums, self.positions))
        return self._chain(batch1, resps)

    def _chain(self, batch1: MaskMutsBatch, resps: pd.DataFrame):
        return self.get_data_type()(batch=batch1.batch,
                                    refseq=batch1.refseq,
                                    muts=batch1.muts,
                                    end5s=batch1.end5s,
                                    mid5s=batch1.mid5s,
                                    mid3s=batch1.mid3s,
                                    end3s=batch1.end3s,
                                    resps=resps,
                                    sanitize=False)