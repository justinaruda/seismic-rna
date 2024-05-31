import os
from logging import getLogger
from pathlib import Path

from click import command

from .ref import get_fasta_path
from ..core import path
from ..core.arg import (docdef,
                        arg_fasta,
                        opt_sim_dir,
                        opt_temp_dir,
                        opt_profile_name,
                        opt_fold_sections_file,
                        opt_fold_coords,
                        opt_fold_primers,
                        opt_fold_constraint,
                        opt_fold_temp,
                        opt_fold_md,
                        opt_fold_mfe,
                        opt_fold_max,
                        opt_fold_percent,
                        opt_keep_temp,
                        opt_force,
                        opt_max_procs,
                        opt_parallel)
from ..core.extern import args_to_cmd, run_cmd
from ..core.parallel import as_list_of_tuples, dispatch, lock_temp_dir
from ..core.rna import renumber_ct
from ..core.seq import DNA, RefSections, Section, parse_fasta, write_fasta
from ..core.write import need_write
from ..fold.rnastructure import make_fold_cmd, retitle_ct_structures

COMMAND = __name__.split(os.path.extsep)[-1]

logger = getLogger(__name__)


def get_ct_path(top: Path, section: Section, profile: str):
    """ Get the path of a connectivity table (CT) file. """
    return path.buildpar(path.RefSeg,
                         path.SectSeg,
                         path.ConnectTableSeg,
                         top=top,
                         ref=section.ref,
                         sect=section.name,
                         profile=profile,
                         ext=path.CT_EXT)


def fold_section(section: Section, *,
                 sim_dir: Path,
                 temp_dir: Path,
                 profile_name: str,
                 fold_constraint: Path | None,
                 fold_temp: float,
                 fold_md: int,
                 fold_mfe: bool,
                 fold_max: int,
                 fold_percent: float,
                 keep_temp: bool,
                 force: bool,
                 n_procs: int):
    ct_sim = get_ct_path(sim_dir, section, profile_name)
    if need_write(ct_sim, force):
        fasta_tmp = get_fasta_path(temp_dir, section.ref)
        ct_tmp = get_ct_path(temp_dir, section, profile_name)
        try:
            # Write a temporary FASTA file for this section only.
            write_fasta(fasta_tmp,
                        [(section.ref, section.seq.tr())],
                        force=force)
            # Predict the RNA structure.
            run_cmd(args_to_cmd(make_fold_cmd(fasta_tmp,
                                              ct_tmp,
                                              dms_file=None,
                                              fold_constraint=fold_constraint,
                                              fold_temp=fold_temp,
                                              fold_md=fold_md,
                                              fold_mfe=fold_mfe,
                                              fold_max=fold_max,
                                              fold_percent=fold_percent,
                                              n_procs=n_procs)))
            # Reformat the CT file title lines so that each is unique.
            retitle_ct_structures(ct_tmp, ct_tmp, force=True)
            # Renumber the CT file so that it has the same numbering
            # scheme as the section, rather than always starting at 1,
            # the latter of which is always output by the Fold program.
            renumber_ct(ct_tmp, ct_sim, section.end5, force=force)
        finally:
            if not keep_temp:
                fasta_tmp.unlink(missing_ok=True)
                if ct_tmp != ct_sim:
                    ct_tmp.unlink(missing_ok=True)
    return ct_sim


@lock_temp_dir
@docdef.auto()
def run(fasta: str, *,
        sim_dir: str,
        temp_dir: str,
        profile_name: str,
        fold_coords: tuple[tuple[str, int, int], ...],
        fold_primers: tuple[tuple[str, DNA, DNA], ...],
        fold_sections_file: str | None,
        fold_constraint: str | None,
        fold_temp: float,
        fold_md: int,
        fold_mfe: bool,
        fold_max: int,
        fold_percent: float,
        keep_temp: bool,
        force: bool,
        max_procs: int,
        parallel: bool):
    try:
        # List the sections.
        sections = RefSections(parse_fasta(Path(fasta), DNA),
                               sects_file=(Path(fold_sections_file)
                                           if fold_sections_file
                                           else None),
                               coords=fold_coords,
                               primers=fold_primers)
    except Exception as error:
        logger.critical(error)
        return list()
    return dispatch(fold_section,
                    max_procs=max_procs,
                    parallel=parallel,
                    hybrid=True,
                    pass_n_procs=True,
                    args=as_list_of_tuples(sections.sections),
                    kwargs=dict(sim_dir=Path(sim_dir),
                                temp_dir=Path(temp_dir),
                                profile_name=profile_name,
                                fold_constraint=(Path(fold_constraint)
                                                 if fold_constraint
                                                 else None),
                                fold_temp=fold_temp,
                                fold_md=fold_md,
                                fold_mfe=fold_mfe,
                                fold_max=fold_max,
                                fold_percent=fold_percent,
                                keep_temp=keep_temp,
                                force=force))


params = [arg_fasta,
          opt_sim_dir,
          opt_temp_dir,
          opt_profile_name,
          opt_fold_sections_file,
          opt_fold_coords,
          opt_fold_primers,
          opt_fold_constraint,
          opt_fold_temp,
          opt_fold_md,
          opt_fold_mfe,
          opt_fold_max,
          opt_fold_percent,
          opt_keep_temp,
          opt_force,
          opt_max_procs,
          opt_parallel]


@command(COMMAND, params=params)
def cli(*args, **kwargs):
    """ Simulate secondary structure(s) a reference sequence. """
    run(*args, **kwargs)
