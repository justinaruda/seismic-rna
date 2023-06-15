from logging import getLogger
from math import inf
from pathlib import Path

from .emalgo import EmClustering
from .metric import find_best_k
from .report import ClustReport
from .write import write_results
from ..core.bitv import UniqMutBits, CloseEmptyBitAccumError
from ..core.parallel import dispatch
from ..mask.load import MaskLoader

logger = getLogger(__name__)


def cluster(call_report: Path, max_clusters: int, n_runs: int, *,
            min_iter: int, max_iter: int, conv_thresh: float, n_procs: int,
            rerun: bool):
    """ Run all processes of clustering reads from one filter. """
    # Load the vector calling report.
    loader = MaskLoader.open(Path(call_report))
    # Check if the clustering report file already exists.
    report_file = ClustReport.build_path(loader.out_dir,
                                         sample=loader.sample,
                                         ref=loader.ref,
                                         sect=loader.sect)
    if rerun or not report_file.is_file():
        logger.info(f"Began EM clustering {loader} with up to k={max_clusters} "
                    f"cluster(s) and n={n_runs} run(s) per number of clusters")
        # Get the unique bit vectors.
        try:
            uniq_muts = loader.get_bit_monolith().get_unique_muts()
        except CloseEmptyBitAccumError:
            # There were no bit vectors.
            raise ValueError(f"Cannot cluster empty {loader}")
        # Run EM clustering for every number of clusters.
        clusts = run_max_clust(loader, uniq_muts, max_clusters, n_runs,
                               min_iter=min_iter, max_iter=max_iter,
                               conv_thresh=conv_thresh, n_procs=n_procs)
        logger.info(f"Ended clustering {loader}: {find_best_k(clusts)} clusters")
        # Output the results of clustering.
        report = ClustReport.from_clusters(clusts, loader, uniq_muts,
                                           max_clusters, n_runs,
                                           min_iter=min_iter,
                                           max_iter=max_iter,
                                           conv_thresh=conv_thresh)
        report.save()
        write_results(loader, clusts)
    else:
        logger.warning(f"File exists: {report_file}")
    # Return the path of the clustering report file.
    return report_file


def run_max_clust(loader: MaskLoader, uniq_muts: UniqMutBits,
                  max_clusters: int, n_runs: int, *,
                  min_iter: int, max_iter: int, conv_thresh: float,
                  n_procs: int):
    """
    Find the optimal number of clusters for EM, up to max_clusters.
    """
    if n_runs < 1:
        logger.warning(f"Number of EM runs must be ≥ 1: setting to 1")
        n_runs = 1
    logger.info(f"Began clustering {loader} with up to {max_clusters} clusters "
                f"and {n_runs} runs per number of clusters")
    # Store the clustering runs for each number of clusters.
    runs: dict[int, list[EmClustering]] = dict()
    # Run clustering for each number of clusters, up to max_clusters.
    while len(runs) < max_clusters:
        # Get the number of clusters, k.
        k = len(runs) + 1
        # Run EM clustering n_runs times with different starting points.
        runs[k] = run_n_clust(loader, uniq_muts, k,
                              n_runs=(n_runs if k > 1 else 1),
                              conv_thresh=(conv_thresh if k > 1 else inf),
                              min_iter=(min_iter * k if k > 1 else 1),
                              max_iter=(max_iter * k if k > 1 else 2),
                              n_procs=n_procs)
        # Find the best (smallest) BIC obtained from clustering.
        best_bic = runs[k][0].bic
        logger.debug(f"The best BIC with {k} cluster(s) is {best_bic}")
        if k > 1:
            prev_bic = runs[k - 1][0].bic
            # Check if the best BIC is better (smaller) than the best
            # BIC from clustering with one cluster fewer.
            if best_bic < prev_bic:
                # If the best BIC is < the previous best BIC, then the
                # current model is the best.
                logger.debug(f"The BIC decreased from {prev_bic} "
                             f"(k = {k - 1}) to {best_bic} "
                             f"(k = {k})")
            else:
                # If the best BIC is ≥ than the previous best BIC, then
                # this model is worse than the previous model. Stop.
                logger.info(f"The BIC increased from {prev_bic} "
                            f"(k = {k - 1}) to {best_bic} "
                            f"(k = {k}): stopping")
                break
    return runs


def run_n_clust(loader: MaskLoader,
                uniq_muts: UniqMutBits,
                n_clusters: int,
                n_runs: int, *,
                min_iter: int, max_iter: int, conv_thresh: float,
                n_procs: int) -> list[EmClustering]:
    """ Run EM with a specific number of clusters. """
    logger.info(f"Began n={n_runs} run(s) of EM with k={n_clusters} cluster(s)")
    # Initialize one EmClustering object for each replicate run.
    runs = [EmClustering(loader, uniq_muts, n_clusters,
                         min_iter=min_iter, max_iter=max_iter,
                         conv_thresh=conv_thresh)
            for _ in range(n_runs)]
    # Run independent replicates of the clustering algorithm.
    runs = dispatch([rep.run for rep in runs], n_procs,
                    parallel=True, pass_n_procs=False)
    logger.info(f"Ended n={n_runs} run(s) of EM with k={n_clusters} cluster(s)")
    # Sort the replicate runs of EM clustering in ascending order
    # by BIC so that the run with the best (smallest) BIC is first.
    return sorted(runs, key=lambda x: x.bic)
