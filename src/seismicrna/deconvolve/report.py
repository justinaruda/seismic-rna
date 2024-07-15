from ..core import path
from ..cluster.io import ClusterIO, ClusterBatchIO
from typing import Any
from ..core.report import (Field,
                           BatchedRefseqReport,
                           SampleF,
                           RefF,
                           SectF,
                           MaxClustsF,
                           DeconvolveClusterMappingF,
                           DeconvolveReadCountsF)


class DeconvolveReport(BatchedRefseqReport, ClusterIO):
        
    @classmethod
    def file_seg_type(cls):
        return path.ClustRepSeg

    @classmethod
    def _batch_types(cls):
        return ClusterBatchIO,
    
    @classmethod
    def fields(cls):
        return [
            # Sample, reference, and section information.
            SampleF,
            RefF,
            SectF,
            # Clustering parameters.
            MaxClustsF,
            DeconvolveClusterMappingF,
            DeconvolveReadCountsF
        ] + super().fields()

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: path.CMD_CLUST_DIR}
    
    @classmethod
    def from_deconvolver(cls, deconvolver):
        """ Create a DeconvolveReport from a Deconvolver object. """
        # Initialize a new DeconvolveReport.
        return cls(sample=deconvolver.sample,
                   ref=deconvolver.ref,
                   sect=deconvolver.sect,
                   max_order=deconvolver.deconvolve_merger.max_order,
                   n_batches=deconvolver.n_batches,
                   checksums={ClusterBatchIO.btype(): ["asdfasdf","1342324","sfgsthsh"]},
                   refseq_checksum="asdfasdg",
                   deconvolution_mapping=deconvolver.mapping_indexes,
                   deconvolve_read_counts=deconvolver.read_counts,
                   began=deconvolver.began,
                   ended=deconvolver.ended)
    