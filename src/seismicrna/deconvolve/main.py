from pathlib import Path

from typing import Iterable

from click import command

from .write import deconvolve
from ..core import path
from ..core.arg import (CMD_DECONVOLVE,
                        arg_input_path,
                        opt_tmp_pfx,
                        opt_keep_tmp,
                        opt_deconvolve_pos_table,
                        opt_deconvolve_abundance_table,
                        opt_brotli_level,
                        opt_max_procs,
                        opt_force)
from ..core.run import run_func
from ..core.task import dispatch
from ..mask.data import load_mask_dataset
from ..core.rel import RelPattern

import itertools

from collections import defaultdict
from ..mask.data import MaskMutsDataset

from ..mask.table import (MaskTable,
                          MaskPositionTableLoader)


from ..core.logs import logger

from .calc import calc_bayes

def load_pos_tables(input_paths: Iterable[str | Path]):
    """ Load position tables. """
    paths = list(input_paths)
    for table_type in [MaskPositionTableLoader]:
        yield from table_type.load_tables(paths)

def _load_and_group_tables(paths: Iterable[str]):
    """
    Load position tables and group them by (reference, section).
    """
    dirs = [path.parent for path in paths]
    tables = load_pos_tables(dirs)
    grouped = defaultdict(list)

    for table in tables:
        ref_sect = (table.ref, table.sect)
        if table not in grouped[ref_sect]:
            grouped[ref_sect].append(table)
        else:
            logger.warning(f"Duplicate table: {table}")

    return grouped

def _get_matching_tables(deconvolve_paths: Iterable[str],
                         no_probe_paths: Iterable[str],
                         only_probe_paths: Iterable[str]):
    """
    Get matching tables grouped by reference and section, based on deconvolve tables.
    """
    # Load deconvolve tables (primary key set) and additional tables
    deconvolve_groups = _load_and_group_tables(deconvolve_paths)
    no_probe_groups = _load_and_group_tables(no_probe_paths)
    only_probe_groups = _load_and_group_tables(only_probe_paths)

    # Filter and populate valid groups in a single pass
    valid_groups = dict()

    for ref_sect, tables in deconvolve_groups.items():
        # Extend with matching no_probe and only_probe tables if they exist
        if ref_sect in no_probe_groups:
            tables.extend(no_probe_groups[ref_sect])
        if ref_sect in only_probe_groups:
            tables.extend(only_probe_groups[ref_sect])

        # Check if we have exactly 3 tables; otherwise, log and skip
        if len(tables) == 3:
            valid_groups[ref_sect] = tables
        else:
            ref, sect = ref_sect
            logger.warning(
                f"Expected 3 tables with reference {repr(ref)} and section {repr(sect)}, "
                f"but got {len(tables)}. Skipping..."
            )

    return valid_groups

@run_func(CMD_DECONVOLVE, with_tmp=True)
def run(deconv_path: tuple[str, ...],
        no_probe_path: tuple[str, ...],
        only_probe_path: tuple[str, ...],
        tup_positions: tuple[Iterable[Iterable[int]], ...],
        tup_patterns: tuple[RelPattern, ...],
        *,
        combinations: Iterable[int] = [1],
        conf_thresh: float,
        norm_edits: bool,
        corr_editing_bias: bool = False,
        n_procs: int,
        strict: bool,
        deconvolve_pos_table: bool,
        deconvolve_abundance_table: bool,
        brotli_level: int,
        max_procs: int,
        force: bool,
        tmp_dir: Path) -> list[Path]:
    """ Cluster reads by mutation pattern. """
    # Find the mask report files.
    path_set = set()
    report_files = list()
    confidences = list()
    all_positions = list()
    all_patterns = list()
    no_probe_samples = list()
    only_probe_samples = list()
    for (deconv_elem,
        no_probe_elem,
        only_probe_elem,
        pattern,
        positions) in zip(deconv_path,
                         no_probe_path,
                         only_probe_path,
                         tup_patterns,
                          tup_positions):
        deconv_reports = path.find_files(deconv_elem,
        load_mask_dataset.report_path_seg_types)
        norm = no_probe_elem is not None and only_probe_elem is not None
        if norm:    
            no_probe_reports = path.find_files(no_probe_elem,
            load_mask_dataset.report_path_seg_types)
            only_probe_reports = path.find_files(only_probe_elem,
            load_mask_dataset.report_path_seg_types)
            table_groups = _get_matching_tables(deconv_reports,
                                                no_probe_reports,
                                                only_probe_reports)
        if deconv_elem not in path_set:
            for file in path.find_files(deconv_elem,
            load_mask_dataset.report_path_seg_types):
                if file not in path_set:
                    report_positions = positions
                    report_indiv_positions = tuple([position[0] for position in positions if len(position)==1])
                    path_set.add(file)
                    if norm:
                        dataset = MaskMutsDataset(file)
                        ref, sect = dataset.ref, dataset.section.name
                        (deconvolve_table,
                         no_probe_table,
                         only_probe_table) = table_groups[ref, sect]
                        bayes = calc_bayes(no_probe_table,
                                            only_probe_table,
                                            pattern)
                        no_probe_samples.append(no_probe_table.sample)
                        only_probe_samples.append(only_probe_table.sample)
                        confidences.append(bayes)
                        conf_positions = bayes[bayes >= conf_thresh].index.get_level_values("Position").values
                        report_indiv_positions += tuple(conf_positions)
                        logger.detail(f"Confident of positions {conf_positions} "
                                      f"for reference: {ref} section: {sect}")
                    for position in report_indiv_positions:
                        report_positions += ((position,),)
                    report_files.append(file)
                    for combination in combinations:
                        report_positions += (tuple([group for group 
                                            in itertools.combinations(
                                                report_indiv_positions, combination)]))
                    all_positions.append(tuple(sorted(set(report_positions), key=lambda x: (len(x), x))))
                    all_patterns.append(pattern)

    arguments = [arg for arg in itertools.zip_longest(report_files,
                                    all_positions,
                                    all_patterns,
                                    confidences,
                                    no_probe_samples,
                                    only_probe_samples)]
    # Cluster each mask dataset.
    return dispatch(deconvolve,
                    max_procs,
                    pass_n_procs=True,
                    args=arguments,
                    kwargs=dict(deconvolve_pos_table=deconvolve_pos_table,
                                conf_thresh=conf_thresh,
                                deconvolve_abundance_table=deconvolve_abundance_table,
                                norm_edits=norm_edits,
                                corr_editing_bias=corr_editing_bias,
                                strict=strict,
                                brotli_level=brotli_level,
                                force=force,
                                tmp_dir=tmp_dir))

params = [
    # Input files
    arg_input_path,
    # Table options
    opt_deconvolve_pos_table,
    opt_deconvolve_abundance_table,
    # Compression
    opt_brotli_level,
    # Parallelization
    opt_max_procs,
    # Effort
    opt_force,
    opt_tmp_pfx,
    opt_keep_tmp,
]


@command(CMD_DECONVOLVE, params=params)
def cli(*args, **kwargs):
    """ Cluster reads by mutation pattern. """
    return run(*args, **kwargs)

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
