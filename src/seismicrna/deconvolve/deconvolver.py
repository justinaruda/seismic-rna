from functools import cached_property, cache

from ..cluster.batch import ClusterMutsBatch
from ..cluster.dataset import ClusterMutsDataset
from ..core.header import list_ks_clusts
from ..mask.batch import MaskMutsBatch
from ..mask.dataset import MaskReadDataset, MaskMutsDataset
from ..core.rel.pattern import RelPattern

from ..mask.table import MaskTabulator, MaskPosTableWriter
from ..cluster.table import ClusterDatasetTabulator, ClusterPosTableWriter

from ..graph.profile import OneRelProfileGraph
from ..graph.corroll import RollingCorrelationGraph
from ..graph.delprof import DeltaProfileGraph

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
        self.deconvolve_dataset = MaskMutsDataset(deconvolve_report)
        self.edited_dataset = MaskMutsDataset(edited_report)
        self.background_dataset = MaskMutsDataset(background_report)
        self.positions = positions
        self.mut_val = mut_val
        self.pattern = pattern
        self.min_reads = min_reads
        self.strict = strict
        self.sample = self.deconvolve_dataset.sample
        self.ref = self.deconvolve_dataset.ref
        self.n_batches = 3
        self.region = self.deconvolve_dataset.region
        self.norm_edits = norm_edits

    def _get_deconvolve_muts_dataset(self):
        self.deconvolve_muts_dataset = DeconvolveMutsDataset(self.positions, 
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
        return ClusterDatasetTabulator(dataset=self.deconvolve_muts_dataset)

    @cached_property
    def deconvolve_tabulator(self):
        return MaskTabulator(self.deconvolve_dataset)

    @cached_property
    def table_writer(self):
        return ClusterPosTableWriter(self.cluster_tabulator)

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
        return {cluster_name: self.deconvolve_muts_dataset.clusters[cluster_num] 
                for cluster_name, cluster_num in self.mapping_indexes.items()}

    @property
    def k_cluster(self):
        index = list(self.mapping_as_clusters.keys())
        values = list(self.mapping_as_clusters.values())
        return pd.Series(values, index=index)

    def graph(self, k=None, clust=None, deconvolved_clusters=None, force=True):
        k_clust_list = list(self.k_cluster[deconvolved_clusters])
        mask_region = self.region.copy()
        mask_region.add_mask(mask_pos=self.positions.ravel(), name='mask-deconvolve')
        self.graph_obj = OneRelDeconvolvedProfileGraph(region=mask_region,
                                                       k=k, 
                                                       clust=clust, 
                                                       k_clust_list=k_clust_list,
                                                       mapping = self.mapping_as_clusters,
                                                       clusters = self.deconvolve_muts_dataset.clusters,
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
        mask_region = self.region.copy()
        mask_region.mask_gu()
        mask_region.add_mask(mask_pos=self.positions.ravel(), name='mask-deconvolve')
        k1, clust1 = self.k_cluster[deconvolved_clust_1]
        k2, clust2 = self.k_cluster[deconvolved_clust_2]
        self.graph_obj = DeconvolvedRollingCorrelationGraph(region=mask_region,
                                                            metric=metric,
                                                            table1=self.table_writer, 
                                                            k1=k1, 
                                                            clust1=clust1, 
                                                            table2=self.table_writer, 
                                                            k2=k2, 
                                                            clust2=clust2, 
                                                            rel="m", 
                                                            window=45, 
                                                            winmin=15, 
                                                            use_ratio=True, 
                                                            quantile=quantile, 
                                                            out_dir=Path("out"))
        self.graph_obj.write_html(force=force)
    
    def graph_delprof(self,
                        deconvolved_clust_1,
                        deconvolved_clust_2,
                        quantile=0.95,
                        force=True):
        mask_region = self.region.copy()
        mask_region.add_mask(mask_pos=self.positions.ravel(), name='mask-deconvolve')
        self.table_writer.region = mask_region
        k1, clust1 = self.k_cluster[deconvolved_clust_1]
        k2, clust2 = self.k_cluster[deconvolved_clust_2]
        self.graph_obj = DeltaProfileGraph(table1=self.table_writer, 
                                           k1=k1, 
                                           clust1=clust1, 
                                           table2=self.table_writer, 
                                           k2=k2, 
                                           clust2=clust2, 
                                           rel="m", 
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
        self.deconvolve_muts_dataset()
        self.began = datetime.now()
        ClusterPosTableWriter(self.cluster_tabulator).write(force=force_write)
        self.ended = datetime.now()
        self.report = DeconvolveReport.from_deconvolver(self)
        self.report.save(top=Path("out"), force=True)
    
        
        
class DeconvolveMutsDataset(ClusterMutsDataset):
    """ Merge deconvolved responsibilities with mutation data. """

    def __init__(self, positions: np.ndarray[int], 
                 mut_val: int, 
                 pattern: RelPattern, 
                 data1, 
                 data2,
                 min_reads=1000, 
                 strict=False,
                 norm_edits=True):
        self.positions = positions
        self.mut_val = mut_val
        self.strict = strict
        self.accum_pattern = pattern
        self.min_reads = min_reads
        self.norm_edits = norm_edits


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
    def region(self):
        # Does this allow modification of the region?
        return self.data1.region
    
    @cached_property
    def max_k(self):
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
        return list_ks_clusts(self.max_k)
    
    @cached_property
    def uniq_reads(self):
        a_pos = (np.array(list(self.region.seq)) != "A").nonzero()[0]
        self.region.add_mask("mask-non-a", a_pos+self.region.coord[0])
        self.region.add_mask("mask-deconvolved", self.positions.ravel())
        only_ag = RelPattern.from_counts(discount=["ac", "at", "ca", "cg", "ct", "ta", "tg", "tc", "ga", "gc", "gt"], count_del=False, count_ins=False)
        
        ((seg_end5s, seg_end3s),
         muts_per_pos,
         batch_to_uniq,
         count_per_uniq) = get_uniq_reads(self.region.unmasked_int,
                                          only_ag,
                                          self.data1.iter_batches(),
                                          )
        
        uniq_reads = UniqReads(self.data1.sample,
                   self.data1.region,
                   self.data1.min_mut_gap,
                   self.data1.quick_unbias,
                   self.data1.quick_unbias_thresh,
                   muts_per_pos,
                   batch_to_uniq,
                   count_per_uniq,
                   seg_end5s=seg_end5s,
                   seg_end3s=seg_end3s)
        
        self.region.remove_mask("mask-non-a")
        self.region.remove_mask("mask-deconvolved")
        return uniq_reads
    
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
                # Pick interregion or union.
                edited = np.interreg1d(edited, muts.get(pos, dict()).get(mut_val, np.array([])))
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

        if norm_edits:
            uniq_ed, uniq_ed_counts = np.unique(self._assign_uniq(batch, edited), return_counts=True)
            uniq_uned, uniq_uned_counts = np.unique(self._assign_uniq(batch, unedited), return_counts=True)

            edited_indexes = np.isin(uniq_ed, uniq_uned)
            unedited_indexes = np.isin(uniq_uned, uniq_ed)
            min_interreg = np.minimum(uniq_ed_counts[edited_indexes],
                                       uniq_uned_counts[unedited_indexes])
            keep_ed = np.zeros_like(edited, dtype=bool)
            keep_uned = np.zeros_like(unedited, dtype=bool)
            batch_to_uniq = self.uniq_reads.batch_to_uniq[batch.batch]
            for read_count, common_uniq_num in zip(min_interreg, uniq_ed[edited_indexes]):
                
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
        batch = self.data1.get_batch(batch_num)
        resps = pd.DataFrame(index=batch.read_nums, 
                             columns=self.clusters, 
                             dtype=int)
        cluster = 1
        for position in self.positions:
            self._get_edited(batch, position)
            resps = self._populate_resps(cluster, batch, resps)
            cluster += 2
            # masked_batch = apply_mask(batch1, positions=np.setdiff1d(batch1.pos_nums, self.positions))
        return self._integrate(batch, resps)

    def _integrate(self, batch: MaskMutsBatch, resps: pd.DataFrame):
        return self.get_data_type()(batch=batch.batch,
                                    refseq=batch.refseq,
                                    muts=batch.muts,
                                    end5s=batch.end5s,
                                    mid5s=batch.mid5s,
                                    mid3s=batch.mid3s,
                                    end3s=batch.end3s,
                                    resps=resps,
                                    sanitize=False)
    
    def iter_batches(self):
        """ Yield each batch. """
        for batch_num in self.batch_nums:
            yield self.get_batch(batch_num)