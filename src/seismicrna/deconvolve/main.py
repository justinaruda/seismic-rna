from pathlib import Path

from typing import Iterable

from click import command

from .write import deconvolve
from ..core import path
from ..core.arg import (MissingOptionError,
                        CMD_DECONVOLVE,
                        arg_input_path,
                        opt_tmp_pfx,
                        opt_keep_tmp,
                        opt_deconv_pos_table,
                        opt_deconv_abundance_table,
                        opt_no_probe_path,
                        opt_only_probe_path,
                        opt_deconv_position,
                        opt_deconv_pattern,
                        opt_deconv_del,
                        opt_deconv_ins,
                        opt_strict,
                        opt_norm_muts,
                        opt_deconv_min_reads,
                        opt_deconv_combos,
                        opt_conf_thresh,
                        opt_deconv_count_mut_conf,
                        opt_brotli_level,
                        opt_max_procs,
                        opt_force)
from ..core.run import run_func
from ..core.task import dispatch
from ..mask.dataset import load_mask_dataset, MaskMutsDataset
from ..core.rel import RelPattern

import itertools

from collections import defaultdict

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
    Load position tables and group them by (reference, region).
    """
    dirs = [path.parent for path in paths]
    tables = load_pos_tables(dirs)
    grouped = defaultdict(list)

    for table in tables:
        ref_reg = (table.ref, table.reg)
        if table not in grouped[ref_reg]:
            grouped[ref_reg].append(table)
        else:
            logger.warning(f"Duplicate table: {table}")

    return grouped

def _get_matching_tables(deconvolve_paths: Iterable[str],
                         no_probe_paths: Iterable[str],
                         only_probe_paths: Iterable[str]):
    """
    Get matching tables grouped by reference and region, based on deconvolve tables.
    """
    # Load deconvolve tables (primary key set) and additional tables
    deconvolve_groups = _load_and_group_tables(deconvolve_paths)
    no_probe_groups = _load_and_group_tables(no_probe_paths)
    only_probe_groups = _load_and_group_tables(only_probe_paths)

    # Filter and populate valid groups in a single pass
    valid_groups = dict()

    for ref_reg, tables in deconvolve_groups.items():
        # Extend with matching no_probe and only_probe tables if they exist
        if ref_reg in no_probe_groups:
            tables.extend(no_probe_groups[ref_reg])
        if ref_reg in only_probe_groups:
            tables.extend(only_probe_groups[ref_reg])

        # Check if we have exactly 3 tables; otherwise, log and skip
        if len(tables) == 3:
            valid_groups[ref_reg] = tables
        else:
            ref, reg = ref_reg
            logger.warning(
                f"Expected 3 tables with reference {repr(ref)} and region {repr(reg)}, "
                f"but got {len(tables)}. Skipping..."
            )

    return valid_groups

@run_func(CMD_DECONVOLVE, with_tmp=True)
def run(input_path: tuple[str, ...],
        *,
        deconv_position: tuple[int, ...] = (),
        deconv_pattern: tuple[str, ...] = (),
        deconv_del: bool,
        deconv_ins: bool,
        no_probe_path: tuple[str, ...],
        only_probe_path: tuple[str, ...],
        deconv_combos: Iterable[int] = [1],
        conf_thresh: float,
        norm_muts: bool,
        strict: bool,
        deconv_count_mut_conf: bool,
        deconv_min_reads: int,
        deconv_pos_table: bool,
        deconv_abundance_table: bool,
        deconv_positions: tuple[Iterable[Iterable[int]], ...] = None,
        deconv_patterns: tuple[RelPattern, ...] = None,
        brotli_level: int,
        max_procs: int,
        force: bool,
        tmp_dir: Path) -> list[Path]:
    """ Cluster reads by mutation pattern. """   
    if deconv_positions is not None:
        deconv_position = deconv_positions
    elif not deconv_position and not conf_thresh:
        logger.error("Missing position(s) to cluster on.")
        raise MissingOptionError("--deconv-position or --conf-thresh is required.")
    elif deconv_position:
        deconv_position = ((deconv_position,),)
    else:
        deconv_position = (tuple(),)
    if deconv_patterns is not None:
        patterns = deconv_patterns
    elif not deconv_pattern:
        logger.error("Missing mutation(s) to cluster on.")
        raise MissingOptionError("--deconv-pattern is required.")
    else:
        all_subs = set(["ag", "at", "ac",
                        "ga", "gt", "gc",
                        "cg", "ct", "ca",
                        "tg", "ta", "tc"])
        deconv_pattern_set = set(deconv_pattern)
        discounts = all_subs - deconv_pattern_set
        patterns = tuple([RelPattern.from_counts(count_del=deconv_del,
                                                 count_ins=deconv_ins,
                                                 discount=discounts,)] * len(input_path))
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
         positions) in zip(input_path,
                           no_probe_path,
                           only_probe_path,
                           patterns,
                           deconv_position):
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
                        ref, reg = dataset.ref, dataset.region.name
                        (deconvolve_table,
                         no_probe_table,
                         only_probe_table) = table_groups.get((ref, reg), (None, None, None))
                        if not deconvolve_table or not no_probe_table or not only_probe_table:
                            continue
                        bayes = calc_bayes(no_probe_table,
                                           only_probe_table,
                                           pattern)
                        no_probe_samples.append(no_probe_table.sample)
                        only_probe_samples.append(only_probe_table.sample)
                        confidences.append(bayes)
                        conf_positions = bayes[bayes >= conf_thresh].index.get_level_values("Position").values
                        report_indiv_positions += tuple(conf_positions)
                        logger.detail(f"Confident of positions {conf_positions} "
                                      f"for reference: {ref} region: {reg}")
                    report_indiv_positions = tuple(sorted(set(report_indiv_positions)))
                    for position in report_indiv_positions:
                        report_positions += ((position,),)
                    report_files.append(file)
                    for combination in deconv_combos:
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
                    kwargs=dict(no_probe_path = no_probe_path,
                                only_probe_path = only_probe_path,
                                deconv_pos_table=deconv_pos_table,
                                deconv_abundance_table=deconv_abundance_table,
                                conf_thresh=conf_thresh,
                                norm_muts=norm_muts,
                                strict=strict,
                                deconv_count_mut_conf=deconv_count_mut_conf,
                                deconv_min_reads=deconv_min_reads,
                                brotli_level=brotli_level,
                                force=force,
                                tmp_dir=tmp_dir))

params = [
    # Input files
    arg_input_path,
    # Deconvolve options
    opt_deconv_position,
    opt_deconv_pattern,
    opt_no_probe_path,
    opt_only_probe_path,
    opt_deconv_del,
    opt_deconv_ins,
    opt_strict,
    opt_norm_muts,
    opt_deconv_min_reads,
    opt_conf_thresh,
    opt_deconv_combos,
    opt_deconv_count_mut_conf,
    # Table options
    opt_deconv_pos_table,
    opt_deconv_abundance_table,
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
