from abc import ABC
from pathlib import Path

import pandas as pd

from .batch import ClusterReadBatch
from .emk import EMRunsK
from ..core import path
from ..core.header import ClustHeader
from ..core.io import ReadBatchIO, RegIO
from ..mask.dataset import MaskMutsDataset


class ClusterIO(RegIO, ABC):

    @classmethod
    def auto_fields(cls):
        return super().auto_fields() | {path.CMD: path.CLUSTER_STEP}


class ClusterBatchIO(ReadBatchIO, ClusterIO, ClusterReadBatch):

    @classmethod
    def file_seg_type(cls):
        return path.ClustBatSeg


class ClusterBatchWriter(object):

    def __init__(self,
                 dataset: MaskMutsDataset,
                 ks: list[EMRunsK],
                 brotli_level: int,
                 top: Path):
        self.dataset = dataset
        # Filter the numbers of clusters, keeping only those with at
        # least one successful run.
        self.ks = [runs for runs in ks if runs.best is not None]
        self.brotli_level = brotli_level
        self.top = top
        self.read_nums = dict()
        self.checksums = list()

    @property
    def ks_written(self):
        return [runs.k for runs in self.ks]

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
            resps = [runs.best.get_resps(mask_batch.batch) for runs in self.ks]
            if resps:
                resps = pd.concat(resps, axis=1)
            else:
                resps = pd.DataFrame(index=self.get_read_nums(mask_batch.batch),
                                     columns=ClustHeader(ks=[]).index)
            batch_file = ClusterBatchIO(sample=self.dataset.sample,
                                        ref=self.dataset.ref,
                                        reg=self.dataset.region.name,
                                        batch=mask_batch.batch,
                                        resps=resps)
            _, checksum = batch_file.save(self.top,
                                          brotli_level=self.brotli_level)
            self.checksums.append(checksum)
