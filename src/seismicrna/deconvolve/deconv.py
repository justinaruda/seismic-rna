from functools import cached_property
from typing import Iterable

import numpy as np
import pandas as pd

from ..cluster.uniq import UniqReads, get_uniq_reads
from ..core.header import ClustHeader
from ..core.logs import logger
from ..core.rel.pattern import RelPattern
from seismicrna.mask.dataset import MaskMutsDataset
from seismicrna.mask.batch import MaskMutsBatch


class DeconvRun(object):
    """ Run deconvolution to cluster the given reads based on mutations at
    the given positions. """

    def __init__(self, 
                 *,
                 dataset: MaskMutsDataset,
                 positions: Iterable[Iterable[int]],
                 pattern: RelPattern,
                 min_reads=1000,
                 strict=False,
                 norm_muts=True):
        """
        Parameters
        ----------
        dataset: MaskMutsDataset
            The dataset to deconvolve.
        positions: Iterable[Iterable[int]]
            The positions/sets of positions to cluster on.
            ((1,),(1,5)) runs clustering twice, first reads  
            are clustered if they have mutations only at position 1
            and second, reads are clustered if they have mutations at 
            both positions 1 and 5.
        pattern: RelPattern
            The relationships to count as mutations.
        min_reads: int = 1000
            Clusters with fewer than this many reads are treated as 
            having zero reads.
        strict: bool = False
            Ensure that reads with a mutation at a specified position 
            are mutated nowhere else. For instance, clustering on position 1 
            without strict=False includes all reads that are mutated at 
            position 1, even if they are also mutated at position 5. 
            With strict=True, if a read is mutated at both position 1 
            and position 5, it is discarded.
        norm_muts: bool = True
            Ensure the distribution of mutations between all other positions are 
            equal between clusters. This setting can discard a large proportion 
            of reads from both clusters, but prevents confounding effects by 
            linked mutations. If the mutations are known to be independent, 
            this option can be disabled.
        """
        self.dataset = dataset
        self.positions = positions
        self.pattern = pattern
        self.min_reads = min_reads
        self.strict = strict
        self.norm_muts = norm_muts
        self._resps = dict()
        # Run deconvolution
        logger.warning(f"Deconvolving {self.dataset.ref} at positions {self.positions}")
        self._run()

    @cached_property
    def num_clusters(self):
        return (len(self.positions) *2) + 1 # One extra column for population average


    @cached_property
    def _clusters(self):
        """ MultiIndex of k and cluster numbers. """
        return ClustHeader(ks=np.array([self.num_clusters])).index
    
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
    def name_to_cluster(self):
        return {cluster_name: self._clusters[cluster_num] 
                for cluster_name, cluster_num in self.mapping_indexes.items()}
    
    @property
    def cluster_to_name(self):
        return {v: k for k, v in self.name_to_cluster.items()}
        
    
    @property
    def read_counts(self):
        batch_count = None
        for batch_num in range(self.dataset.num_batches):
            resps = self.get_resps(batch_num)
            if batch_count is None:
                batch_count = resps.sum(axis=0)
            else:
                batch_count += resps.sum(axis=0)
        batch_count.index = batch_count.index.to_flat_index()
        read_count_dict = batch_count.rename(self.cluster_to_name).to_dict()
        return read_count_dict

    
    @cached_property
    def uniq_reads(self):
        a_pos = (np.array(list(self.dataset.region.seq)) != "A").nonzero()[0]
        self.dataset.region.add_mask("mask-non-a", a_pos+self.dataset.region.coord[0])
        self.current_positions = np.asarray(self.current_positions)
        self.dataset.region.add_mask("mask-deconvolved", self.current_positions.ravel())
        only_ag = RelPattern.from_counts(discount=["ac", "at", "ca", "cg", "ct", "ta", "tg", "tc", "ga", "gc", "gt"], count_del=False, count_ins=False)
        
        ((seg_end5s, seg_end3s),
         muts_per_pos,
         batch_to_uniq,
         count_per_uniq) = get_uniq_reads(self.dataset.region.unmasked_int,
                                          only_ag,
                                          self.dataset.iter_batches(),
                                          )
        
        uniq_reads = UniqReads(self.dataset.sample,
                   self.dataset.region,
                   self.dataset.min_mut_gap,
                   self.dataset.quick_unbias,
                   self.dataset.quick_unbias_thresh,
                   muts_per_pos,
                   batch_to_uniq,
                   count_per_uniq,
                   seg_end5s=seg_end5s,
                   seg_end3s=seg_end3s)
        
        self.dataset.region.remove_mask("mask-non-a")
        self.dataset.region.remove_mask("mask-deconvolved")
        return uniq_reads
    
    def _assign_uniq(self, batch, read_nums):
        return self.uniq_reads.batch_to_uniq[batch.batch][read_nums]
    
    def _get_edited(self, batch):
        ref = self.dataset.refseq
        strict = self.strict
        norm_muts = self.norm_muts
        muts = batch.muts
        edited = None
        unedited = None
        edited_partial = set()
        interregion = True
        for pos in self.current_positions:
            ref_base = ref[pos-1]
            if edited is None:
                edited = set()
                for rel, read_nums in muts.get(pos, dict()).items():
                    if all(self.pattern.fits(ref_base, rel)):
                        edited.update(read_nums)
            else:
                for rel, read_nums in muts.get(pos, dict()).items():
                    if all(self.pattern.fits(ref_base, rel)):
                        # Pick interregion or union.
                        if interregion:
                            read_nums = set(read_nums)
                            edited_partial = edited_partial | (edited ^ read_nums)
                            edited &= read_nums
                        else:
                            edited |= read_nums
        edited -= edited_partial
        unedited = set(batch.read_nums) - edited
        if strict:
            remove = set()
            for pos in muts:
                if pos in self.current_positions:
                    continue
                muts_at_pos = set()
                for rel, read_nums in muts.get(pos, dict()).items():
                    if all(self.pattern.fits(ref_base, rel)):
                            muts_at_pos.update(read_nums)
                not_mut_at_pos = set(batch.read_nums) - muts_at_pos
                edited_at_pos = edited - not_mut_at_pos
                remove |= edited_at_pos
            edited -= remove
            unedited -= remove
        
        edited = np.fromiter(edited, int, len(edited))
        unedited = np.fromiter(unedited, int, len(unedited))

        if norm_muts:
            
            uniq_ed, uniq_ed_counts = np.unique(self._assign_uniq(batch, edited), return_counts=True)
            uniq_uned, uniq_uned_counts = np.unique(self._assign_uniq(batch, unedited), return_counts=True)

            edited_indexes = np.isin(uniq_ed, uniq_uned)
            unedited_indexes = np.isin(uniq_uned, uniq_ed)
            min_intersect = np.minimum(uniq_ed_counts[edited_indexes],
                                       uniq_uned_counts[unedited_indexes])
            
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
        resps_matrix = np.zeros((batch.num_reads, 2), dtype=int)
        resps_matrix[batch.read_indexes[self.edited], 0] = 1
        resps_matrix[batch.read_indexes[self.unedited], 1] = 1   
        resps.loc[:,resps.columns[cluster]] = resps_matrix[:, 0]
        resps.loc[:,resps.columns[cluster+1]] = resps_matrix[:, 1]
        return resps

    def _calc_resps(self, batch_num: int):
        batch = self.dataset.get_batch(batch_num)
        resps = pd.DataFrame(index=batch.read_nums, 
                             columns=self._clusters, 
                             dtype=int)
        cluster = 1
        for position in self.positions:
            self.current_positions = position
            self._get_edited(batch)
            resps = self._populate_resps(cluster, batch, resps)
            cluster += 2
            # if hasattr(self, "uniq_reads"):
            #     del self.uniq_reads
        return resps
    
    def _run(self):
        for batch_num in range(self.dataset.num_batches):
            self._resps[batch_num] = self._calc_resps(batch_num)
        read_counts = np.zeros(self.num_clusters, dtype=int)
        for resp_df in self._resps.values():
            read_counts += resp_df.sum()
        mask = read_counts < self.min_reads
        for resp_df in self._resps.values():
            resp_df.loc[:, mask] = 0
        
    
    def get_resps(self, batch_num):
        if batch_num not in self._resps:
            logger.error(f"""Could not find responsbilities
                         for batch {batch_num}""")
        return self._resps.get(batch_num)


    def __str__(self):
        return f"{type(self).__name__} {self.uniq_reads}, {self.k} cluster(s)"

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
