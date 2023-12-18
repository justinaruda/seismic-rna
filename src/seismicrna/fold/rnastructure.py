"""
Struct -- RNAstructure Module

Wrapper around RNAstructure from the Mathews Lab at U of Rochester:
https://rna.urmc.rochester.edu/RNAstructure.html
"""

import re
from logging import getLogger
from pathlib import Path

from ..core import path
from ..core.extern import (RNASTRUCTURE_CT2DOT_CMD,
                           RNASTRUCTURE_DOT2CT_CMD,
                           RNASTRUCTURE_FOLD_CMD,
                           args_to_cmd,
                           run_cmd)
from ..core.rna import RNAProfile, renumber_ct
from ..core.write import need_write

logger = getLogger(__name__)


def fold(rna: RNAProfile, *,
         out_dir: Path,
         temp_dir: Path,
         keep_temp: bool,
         force: bool):
    """ Run the 'Fold' program of RNAstructure. """
    ct_file = rna.get_ct_file(out_dir)
    if need_write(ct_file, force):
        cmd = [RNASTRUCTURE_FOLD_CMD]
        # Write the DMS reactivities file for the RNA.
        cmd.extend(["--DMS", dms_file := rna.to_dms(temp_dir)])
        # Write a temporary FASTA file for the RNA.
        cmd.append(fasta := rna.to_fasta(temp_dir))
        # Determine the path of the temporary CT file.
        cmd.append(ct_temp := rna.get_ct_file(temp_dir))
        try:
            # Run the command.
            run_cmd(args_to_cmd(cmd))
            # Renumber the CT file so that it has the same numbering
            # scheme as the section, rather than always starting at 1,
            # the latter of which is always output by the Fold program.
            renumber_ct(ct_temp, ct_file, rna.section.end5, force)
        finally:
            if not keep_temp:
                # Delete the temporary files.
                fasta.unlink(missing_ok=True)
                dms_file.unlink(missing_ok=True)
                ct_temp.unlink(missing_ok=True)
    return ct_file


def ct2dot(ct_file: Path, number: int | str = "all"):
    """ Make a dot-bracket (DB) file of a connectivity-table (CT) file.

    Parameters
    ----------
    ct_file: pathlib.Path
        Path to the CT file.
    number: int | str = "all"
        Number of the structure to convert, or "all" to convert all.

    Returns
    -------
    pathlib.Path
        Path to the DB file.
    """
    dot_file = ct_file.with_suffix(path.DOT_EXT)
    cmd = [RNASTRUCTURE_CT2DOT_CMD, ct_file, number, dot_file]
    run_cmd(args_to_cmd(cmd))
    return dot_file


def dot2ct(db_file: Path):
    """ Make a connectivity-table (CT) file of a dot-bracket (DB) file.

    Parameters
    ----------
    db_file: pathlib.Path
        Path to the DB file.

    Returns
    -------
    pathlib.Path
        Path to the CT file.
    """
    ct_file = db_file.with_suffix(path.CT_EXT)
    cmd = [RNASTRUCTURE_DOT2CT_CMD, db_file, ct_file]
    run_cmd(args_to_cmd(cmd))
    return ct_file


def parse_energy(line: str):
    """ Parse the predicted free energy of folding from a line.

    Parameters
    ----------
    line: str
        Line from which to parse the energy.

    Returns
    -------
    float
        Free energy of folding.
    """
    if not (match := re.search(f"ENERGY = (-?[0-9.]+)", line)):
        raise ValueError(f"Failed to parse energy from line {repr(line)}")
    return float(match.groups()[0])


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
