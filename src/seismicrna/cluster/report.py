from .compare import RunOrderResults, find_best_order
from ..mask.data import MaskLoader
from ..core import path
from ..core.bitvect import UniqMutBits
from ..core.clicmd import CMD_CLUST
from ..core.report import BatchedReport


class ClustReport(BatchedReport):
    __slots__ = (
        # Sample, reference, and section information.
        "sample", "ref", "sect", "end5", "end3", "n_uniq_reads",
        # Clustering parameters.
        "max_order", "num_runs", "min_iter", "max_iter", "conv_thresh",
        # Batch information.
        "checksums", "n_batches",
        # Clustering results.
        "converged", "log_likes", "log_like_mean", "log_like_std", "var_info",
        "bic", "best_order",
    )

    @classmethod
    def path_segs(cls):
        return (path.SampSeg, path.CmdSeg, path.RefSeg, path.SectSeg,
                path.ClustRepSeg)

    @classmethod
    def auto_fields(cls):
        return {**super().auto_fields(), path.CMD: CMD_CLUST}

    @classmethod
    def get_batch_seg(cls):
        return path.ClustBatSeg

    @classmethod
    def from_clusters(cls, /,
                      ord_runs: dict[int, RunOrderResults],
                      loader: MaskLoader,
                      uniq_muts: UniqMutBits,
                      max_order: int,
                      num_runs: int, *,
                      min_iter: int,
                      max_iter: int,
                      conv_thresh: float,
                      checksums: list[str]):
        """ Create a ClusterReport from EmClustering objects. """
        # Initialize a new ClusterReport.
        return cls(out_dir=loader.out_dir,
                   sample=loader.sample,
                   ref=loader.ref,
                   sect=loader.sect,
                   end5=loader.end5,
                   end3=loader.end3,
                   n_uniq_reads=uniq_muts.n_uniq,
                   max_order=max_order,
                   num_runs=num_runs,
                   min_iter=min_iter,
                   max_iter=max_iter,
                   conv_thresh=conv_thresh,
                   checksums=checksums,
                   n_batches=len(checksums),
                   converged={order: runs.converged
                              for order, runs in ord_runs.items()},
                   log_likes={order: runs.log_likes
                              for order, runs in ord_runs.items()},
                   log_like_mean={order: runs.log_like_mean
                                  for order, runs in ord_runs.items()},
                   log_like_std={order: runs.log_like_std
                                 for order, runs in ord_runs.items()},
                   var_info={order: runs.var_info
                             for order, runs in ord_runs.items()},
                   bic={order: runs.best.bic
                        for order, runs in ord_runs.items()},
                   best_order=find_best_order(ord_runs))

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
