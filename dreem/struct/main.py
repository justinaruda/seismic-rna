from logging import getLogger
from pathlib import Path

from click import command

from .rnastructure import fold, ct2dot
from ..core import docdef, path
from ..core.cli import (opt_temp_dir, opt_save_temp, opt_table,
                        opt_fasta, opt_library,
                        opt_coords, opt_primers, opt_primer_gap,
                        opt_dms_quantile,
                        opt_max_procs, opt_parallel, opt_rerun)
from ..core.dependencies import check_rnastructure_exists
from ..core.parallel import as_list_of_tuples, dispatch, lock_temp_dir
from ..core.rna import RnaProfile
from ..core.sect import RefSections, Section, encode_primers
from ..core.seq import parse_fasta
from ..table.load import load, MaskPosTableLoader, ClusterPosTableLoader

logger = getLogger(__name__)

params = [
    opt_table,
    opt_fasta,
    opt_library,
    opt_coords,
    opt_primers,
    opt_primer_gap,
    opt_dms_quantile,
    opt_temp_dir,
    opt_save_temp,
    opt_max_procs,
    opt_parallel,
    opt_rerun,
]


@command(path.MOD_STRUCT, params=params)
def cli(*args, **kwargs):
    """ Predict the structure(s) of an RNA using mutation rates from the
    individual clusters or the ensemble average ('mask' step). """
    return run(*args, **kwargs)


@lock_temp_dir
@docdef.auto()
def run(table: tuple[str, ...],
        *,
        fasta: str,
        library: str | None,
        coords: tuple[tuple[str, int, int], ...],
        primers: tuple[tuple[str, str, str], ...],
        primer_gap: int,
        dms_quantile: float,
        temp_dir: str,
        save_temp: bool,
        max_procs: int,
        parallel: bool,
        rerun: bool):
    """
    Run the structure module.
    """
    check_rnastructure_exists()

    if not fasta:
        logger.critical(f"No FASTA file given to {path.MOD_STRUCT}")
        return list()

    # Get the sections for every reference sequence.
    ref_sections = RefSections(parse_fasta(Path(fasta)),
                               library=Path(library) if library else None,
                               coords=coords,
                               primers=encode_primers(primers),
                               primer_gap=primer_gap)
    # Initialize the table loaders.
    files = path.find_files_multi(map(Path, table), [path.MutTabSeg])
    loaders = [loader for loader in dispatch(load, max_procs, parallel,
                                             args=as_list_of_tuples(files),
                                             pass_n_procs=False)
               if isinstance(loader, (MaskPosTableLoader,
                                      ClusterPosTableLoader))]
    # Fold the RNA profiles.
    return dispatch(fold_rna, max_procs, parallel,
                    args=[(loader, ref_sections.list(loader.ref))
                          for loader in loaders],
                    kwargs=dict(temp_dir=Path(temp_dir), save_temp=save_temp,
                                dms_quantile=dms_quantile, rerun=rerun),
                    pass_n_procs=True)


def fold_rna(loader: MaskPosTableLoader | ClusterPosTableLoader,
             sections: list[Section], n_procs: int, **kwargs):
    """ Fold an RNA molecule from one table of reactivities. """
    return dispatch(fold_profile, n_procs, parallel=True,
                    args=[(profile,)
                          for profile in loader.iter_profiles(sections)],
                    kwargs=dict(out_dir=loader.out_dir, **kwargs),
                    pass_n_procs=False)


def fold_profile(rna: RnaProfile, out_dir: Path, dms_quantile: float, **kwargs):
    """ Fold a section of an RNA from one mutational profile. """
    ct_file = fold(rna, out_dir=out_dir, dms_quantile=dms_quantile, **kwargs)
    dot_file = ct2dot(ct_file)
    varnac_file = rna.to_varnac(out_dir, dms_quantile)
    return ct_file, dot_file, varnac_file