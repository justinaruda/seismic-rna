from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import pandas as pd

from .index import RB_INDEX_NAMES
from ..array import calc_inverse, get_length
from ..types import fit_uint_type


class ReadBatch(ABC):
    """ Batch of reads. """

    def __init__(self, *, batch: int):
        self.batch = batch
        self.masked_read_nums = None

    @cached_property
    @abstractmethod
    def max_read(self) -> int:
        """ Maximum possible value for a read index. """

    @cached_property
    @abstractmethod
    def num_reads(self) -> int | pd.Series:
        """ Number of reads. """

    @cached_property
    @abstractmethod
    def read_nums(self) -> np.ndarray:
        """ Read numbers. """

    @cached_property
    def read_dtype(self):
        """ Data type for read numbers. """
        return fit_uint_type(max(self.max_read, 0))

    @cached_property
    @abstractmethod
    def read_indexes(self) -> np.ndarray:
        """ Map each read number to its index in self.read_nums. """

    @cached_property
    def batch_read_index(self):
        """ MultiIndex of the batch number and read numbers. """
        return pd.MultiIndex.from_arrays(
            [np.broadcast_to(self.batch,
                             get_length(self.read_nums, "read_nums")),
             self.read_nums],
            names=RB_INDEX_NAMES)

    @property
    def masked_reads_bool(self):
        masked_reads_bool = np.zeros_like(self.read_nums, dtype=bool)
        if self.masked_read_nums is not None:
            masked_reads_bool[self.read_indexes[self.masked_read_nums]] = 1
        return masked_reads_bool

    def __str__(self):
        return (f"{type(self).__name__} {self.batch} with "
                f"{get_length(self.read_nums, 'read_nums')} reads")


class AllReadBatch(ReadBatch, ABC):

    @cached_property
    def read_nums(self):
        return np.arange(self.num_reads, dtype=self.read_dtype)

    @cached_property
    def max_read(self):
        return self.num_reads - 1

    @cached_property
    def read_indexes(self):
        return self.read_nums


class PartialReadBatch(ReadBatch, ABC):

    @cached_property
    def max_read(self):
        return self.read_nums.max(initial=0)

    @cached_property
    def read_indexes(self):
        return calc_inverse(self.read_nums, what="read_nums", verify=False)

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
