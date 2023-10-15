from abc import ABC

from ..core import path
from ..core.batch import MaskReadBatch
from ..core.io import ReadBatchIO, SectIO


class MaskIO(SectIO, ABC):

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: path.DIR_MASK}


class MaskReadBatchIO(ReadBatchIO, MaskIO, MaskReadBatch):

    @classmethod
    def file_seg_type(cls):
        return path.MaskBatSeg