from datetime import datetime

from .emk import EMRunsK, find_best_k
from .io import ClusterIO, ClusterBatchIO
from .uniq import UniqReads
from ..core import path
from ..core.join import JoinReport
from ..core.report import (BatchedReport,
                           SampleF,
                           RefF,
                           RegF,
                           NumUniqReadKeptF,
                           MinClustsF,
                           MaxClustsF,
                           ClustNumRunsF,
                           TryAllKsF,
                           WriteAllKsF,
                           MaxPearsonRunF,
                           MinMARCDRunF,
                           JackpotF,
                           JackpotConfLevelF,
                           MaxJackpotQuotientF,
                           MaxLogLikeVsBestF,
                           MinPearsonVsBestF,
                           MaxMARCDVsBestF,
                           MinIterClustF,
                           MaxIterClustF,
                           ClustConvThreshF,
                           EMKPassingF,
                           BestKF,
                           KsWrittenF,
                           JoinedRegionsF,
                           JoinedClustersF)


class ClusterReport(BatchedReport, ClusterIO):

    @classmethod
    def file_seg_type(cls):
        return path.ClustRepSeg

    @classmethod
    def _batch_types(cls):
        return [ClusterBatchIO]

    @classmethod
    def fields(cls):
        return [
            # Sample, reference, and region information.
            SampleF,
            RefF,
            RegF,
            NumUniqReadKeptF,
            # Clustering parameters.
            MinClustsF,
            MaxClustsF,
            TryAllKsF,
            WriteAllKsF,
            MaxPearsonRunF,
            MinMARCDRunF,
            JackpotF,
            JackpotConfLevelF,
            MaxJackpotQuotientF,
            MaxLogLikeVsBestF,
            MinPearsonVsBestF,
            MaxMARCDVsBestF,
            ClustNumRunsF,
            MinIterClustF,
            MaxIterClustF,
            ClustConvThreshF,
            # Clustering results.
            EMKPassingF,
            BestKF,
            KsWrittenF,
        ] + super().fields()

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: path.CLUSTER_STEP}

    @classmethod
    def from_clusters(cls,
                      ks: list[EMRunsK],
                      uniq_reads: UniqReads, *,
                      min_clusters: int,
                      max_clusters: int,
                      em_runs: int,
                      try_all_ks: bool,
                      write_all_ks: bool,
                      min_marcd_run: float,
                      max_pearson_run: float,
                      jackpot: bool,
                      jackpot_conf_level: float,
                      max_jackpot_quotient: float,
                      max_loglike_vs_best: float,
                      min_pearson_vs_best: float,
                      max_marcd_vs_best: float,
                      min_iter: int,
                      max_iter: int,
                      em_thresh: float,
                      ks_written: list[int],
                      checksums: list[str],
                      began: datetime,
                      ended: datetime):
        """ Create a ClusterReport from EmClustering objects. """
        return cls(sample=uniq_reads.sample,
                   ref=uniq_reads.ref,
                   reg=uniq_reads.region.name,
                   n_uniq_reads=uniq_reads.num_uniq,
                   min_clusters=min_clusters,
                   max_clusters=max_clusters,
                   em_runs=em_runs,
                   try_all_ks=try_all_ks,
                   write_all_ks=write_all_ks,
                   min_marcd_run=min_marcd_run,
                   max_pearson_run=max_pearson_run,
                   jackpot=jackpot,
                   jackpot_conf_level=jackpot_conf_level,
                   max_jackpot_quotient=max_jackpot_quotient,
                   max_loglike_vs_best=max_loglike_vs_best,
                   min_pearson_vs_best=min_pearson_vs_best,
                   max_marcd_vs_best=max_marcd_vs_best,
                   min_em_iter=min_iter,
                   max_em_iter=max_iter,
                   em_thresh=em_thresh,
                   checksums={ClusterBatchIO.btype(): checksums},
                   n_batches=len(checksums),
                   em_k_passing={runs.k: runs.passing() for runs in ks},
                   best_k=find_best_k(ks),
                   ks_written=ks_written,
                   began=began,
                   ended=ended)


class JoinClusterReport(JoinReport):

    @classmethod
    def file_seg_type(cls):
        return path.ClustRepSeg

    @classmethod
    def fields(cls):
        return [
            # Sample and reference.
            SampleF,
            RefF,
            RegF,
            # Joined data.
            JoinedRegionsF,
            JoinedClustersF,
        ] + super().fields()

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: path.CLUSTER_STEP}
