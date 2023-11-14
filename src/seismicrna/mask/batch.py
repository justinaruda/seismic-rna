from abc import ABC
from typing import Iterable

import numpy as np

from ..core.batch import (RefseqMutsBatch,
                          PartialMutsBatch,
                          PartialReadBatch,
                          sanitize_pos)


class MaskReadBatch(PartialReadBatch):

    def __init__(self, *, read_nums: np.ndarray, **kwargs):
        self._read_nums = read_nums
        super().__init__(**kwargs)

    @property
    def read_nums(self):
        return self._read_nums


class MaskMutsBatch(MaskReadBatch, RefseqMutsBatch, PartialMutsBatch, ABC):
    pass


def apply_mask(batch: RefseqMutsBatch,
               reads: Iterable[int] | None = None,
               positions: Iterable[int] | None = None,
               return_inverse_reads: bool = False,
               return_inverse_positions: bool = False):
    if return_inverse_reads or return_inverse_positions:
        if return_inverse_reads:
            if reads is None or reads.size == 0:
                raise AttributeError("To return inverse reads, \
                                     at least 1 read must be masked.")
        if return_inverse_positions:
            if positions is None or positions.size == 0:
                raise AttributeError("To return inverse positions, \
                                     at least 1 position must be masked.")
        inverse = dict()
    # Sanitize and validate positions
    if positions is not None:
        positions = sanitize_pos(positions, batch.max_pos)
    else:
        positions = batch.pos_nums
    if return_inverse_positions:
        inverse_positions = np.setdiff1d(batch.pos_nums, positions)
    muts = dict()
    for pos in positions:
        muts[pos] = dict()
        for mut, pos_mut_reads in batch.muts.get(pos, dict()).items():
            selected_reads = (np.intersect1d(pos_mut_reads,
                                            reads,
                                            assume_unique=True)
                                              if reads
                                              is not None
                                              else pos_mut_reads)
            muts[pos][mut] = selected_reads
    # Handle inverse reads.
    if return_inverse_reads:
        for pos in (inverse_positions 
                    if return_inverse_positions else positions):
            inverse[pos] = dict()
            for mut, pos_mut_reads in batch.muts.get(pos, dict()).items():
                inverse[pos][mut] = np.setdiff1d(pos_mut_reads,
                                                 reads,
                                                 assume_unique=True)
    elif return_inverse_positions:
        for pos in inverse_positions:
            if pos in batch.muts:
                inverse[pos] = batch.muts[pos]
    if reads is not None:
        read_nums = np.asarray(reads, dtype=batch.read_dtype)
        read_indexes = batch.read_indexes[read_nums]
        end5s = batch.end5s[read_indexes]
        mid5s = batch.mid5s[read_indexes]
        mid3s = batch.mid3s[read_indexes]
        end3s = batch.end3s[read_indexes]
        # If inverting reads, remove selected reads from attributes.
        if return_inverse_reads:
            inv_read_nums = np.delete(batch.read_nums, read_indexes)
            inv_end5s = np.delete(batch.end5s, read_indexes)
            inv_mid5s = np.delete(batch.mid5s, read_indexes)
            inv_mid3s = np.delete(batch.mid3s, read_indexes)
            inv_end3s = np.delete(batch.end3s, read_indexes)
        elif return_inverse_positions:
            inv_read_nums = read_nums
            inv_end5s = end5s
            inv_mid5s = mid5s
            inv_mid3s = mid3s
            inv_end3s = end3s
    else:
        read_nums = batch.read_nums
        end5s = batch.end5s
        mid5s = batch.mid5s
        mid3s = batch.mid3s
        end3s = batch.end3s
        if return_inverse_positions:
            inv_read_nums = read_nums
            inv_end5s = end5s
            inv_mid5s = mid5s
            inv_mid3s = mid3s
            inv_end3s = end3s
    if return_inverse_reads or return_inverse_positions:
        return (MaskMutsBatch(batch=batch.batch,
                              refseq=batch.refseq,
                              muts=muts,
                              read_nums=read_nums,
                              end5s=end5s,
                              mid5s=mid5s,
                              mid3s=mid3s,
                              end3s=end3s,
                              sanitize=False),
                MaskMutsBatch(batch=batch.batch,
                              refseq=batch.refseq,
                              muts=inverse,
                              read_nums=inv_read_nums,
                              end5s=inv_end5s,
                              mid5s=inv_mid5s,
                              mid3s=inv_mid3s,
                              end3s=inv_end3s,
                              sanitize=False))
    
    return MaskMutsBatch(batch=batch.batch,
                         refseq=batch.refseq,
                         muts=muts,
                         read_nums=read_nums,
                         end5s=end5s,
                         mid5s=mid5s,
                         mid3s=mid3s,
                         end3s=end3s,
                         sanitize=False)

########################################################################
#                                                                      #
# Copyright ©2023, the Rouskin Lab.                                    #
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
