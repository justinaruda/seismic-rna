from abc import ABC
from pathlib import Path
from typing import Iterable

from datetime import datetime

import pandas as pd

from .batch import DeconvolveReadBatch
from ..core import path
from ..core.header import ClustHeader
from ..core.io import ReadBatchIO, RegIO
from ..mask.dataset import MaskMutsDataset
from .deconv import DeconvRun



class DeconvolveIO(RegIO, ABC):

    @classmethod
    def auto_fields(cls):
        return super().auto_fields() | {path.CMD: path.DECONVOLVE_STEP}


class DeconvolveBatchIO(ReadBatchIO, DeconvolveIO, DeconvolveReadBatch):

    @classmethod
    def file_seg_type(cls):
        return path.DeconvBatSeg


class DeconvolveBatchWriter(object):

    def __init__(self,
                 deconv_run: DeconvRun,
                 brotli_level: int,
                 top: Path):
        self.deconv_run = deconv_run
        self.dataset = self.deconv_run.dataset
        self.brotli_level = brotli_level
        self.top = top
        self.read_nums = dict()
        self.checksums = list()


    def get_read_nums(self, batch_num: int):
        """ Get the read numbers for one batch. """
        if (nums := self.read_nums.get(batch_num)) is not None:
            return nums
        nums = self.dataset.get_batch(batch_num).read_nums
        self.read_nums[batch_num] = nums
        return nums

    def write_batches(self):
        """ Save the batches. """
        for mask_batch in self.dataset.iter_batches():
            resps = self.deconv_run.get_resps(mask_batch.batch)
            batch_file = DeconvolveBatchIO(sample=self.dataset.sample,
                                        ref=self.dataset.ref,
                                        reg=self.dataset.region.name,
                                        batch=mask_batch.batch,
                                        resps=resps)
            _, checksum = batch_file.save(self.top,
                                          brotli_level=self.brotli_level)
            self.checksums.append(checksum)

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
