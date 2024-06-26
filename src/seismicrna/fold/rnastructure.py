"""
Struct -- RNAstructure Module

Wrapper around RNAstructure from the Mathews Lab at U of Rochester:
https://rna.urmc.rochester.edu/RNAstructure.html
"""

import os
import re
from logging import getLogger
from pathlib import Path

from ..core import path
from ..core.extern import (RNASTRUCTURE_CT2DOT_CMD,
                           RNASTRUCTURE_DOT2CT_CMD,
                           RNASTRUCTURE_FOLD_CMD,
                           RNASTRUCTURE_FOLD_SMP_CMD,
                           args_to_cmd,
                           run_cmd)
from ..core.rna import RNAProfile, renumber_ct
from ..core.write import need_write, write_mode

logger = getLogger(__name__)

FOLD_SMP_NUM_THREADS = "OMP_NUM_THREADS"
DATAPATH = "DATAPATH"
DATAPATH_FILES = """
autodetect.txt
average_ensemble_defect.model
b-test.coaxial.dg
b-test.coaxial.dh
b-test.coaxstack.dg
b-test.coaxstack.dh
b-test.dangle.dg
b-test.dangle.dh
b-test.dynalignmiscloop.dg
b-test.hexaloop.dg
b-test.hexaloop.dh
b-test.int11.dg
b-test.int11.dh
b-test.int21.dg
b-test.int21.dh
b-test.int22.dg
b-test.int22.dh
b-test.loop.dg
b-test.miscloop.dg
b-test.miscloop.dh
b-test.specification.dat
b-test.stack.dg
b-test.stack.dh
b-test.tloop.dg
b-test.tloop.dh
b-test.triloop.dg
b-test.triloop.dh
b-test.tstack.dg
b-test.tstack.dh
b-test.tstackcoax.dg
b-test.tstackcoax.dh
b-test.tstackh.dg
b-test.tstackh.dh
b-test.tstacki.dg
b-test.tstacki.dh
b-test.tstacki1n.dg
b-test.tstacki1n.dh
b-test.tstacki23.dg
b-test.tstacki23.dh
b-test.tstackm.dg
b-test.tstackm.dh
coaxial.dat
coaxial.dh
coaxstack.dat
coaxstack.dh
dangle.dat
dangle.dh
data_assemble_training_Multifind_predict_ensemble_z_final_svmformat.model
description.txt
design.DNA.Helices.dat
design.DNA.Loops.dat
design.RNA.Helices.dat
design.RNA.Loops.dat
dists
dna.coaxial.dg
dna.coaxial.dh
dna.coaxstack.dg
dna.coaxstack.dh
dna.dangle.dg
dna.dangle.dh
dna.dynalignmiscloop.dg
dna.dynalignmiscloop.dh
dna.hexaloop.dg
dna.hexaloop.dh
dna.int11.dg
dna.int11.dh
dna.int21.dg
dna.int21.dh
dna.int22.dg
dna.int22.dh
dna.loop.dg
dna.loop.dh
dna.miscloop.dg
dna.miscloop.dh
dna.specification.dat
dna.stack.dg
dna.stack.dh
dna.tloop.dg
dna.tloop.dh
dna.triloop.dg
dna.triloop.dh
dna.tstack.dg
dna.tstack.dh
dna.tstackcoax.dg
dna.tstackcoax.dh
dna.tstackh.dg
dna.tstackh.dh
dna.tstacki.dg
dna.tstacki.dh
dna.tstacki1n.dg
dna.tstacki1n.dh
dna.tstacki23.dg
dna.tstacki23.dh
dna.tstackm.dg
dna.tstackm.dh
dnacoaxial.dat
dnacoaxial.dh
dnacoaxstack.dat
dnacoaxstack.dh
dnadangle.dat
dnadangle.dh
dnadynalignmiscloop.dat
dnadynalignmiscloop.dh
dnahexaloop.dat
dnahexaloop.dh
dnaint11.dat
dnaint11.dh
dnaint21.dat
dnaint21.dh
dnaint22.dat
dnaint22.dh
dnaloop.dat
dnaloop.dh
dnamiscloop.dat
dnamiscloop.dh
dnastack.dat
dnastack.dh
dnatloop.dat
dnatloop.dh
dnatriloop.dat
dnatriloop.dh
dnatstack.dat
dnatstack.dh
dnatstackcoax.dat
dnatstackcoax.dh
dnatstackh.dat
dnatstackh.dh
dnatstacki.dat
dnatstacki.dh
dnatstacki1n.dat
dnatstacki1n.dh
dnatstacki23.dat
dnatstacki23.dh
dnatstackm.dat
dnatstackm.dh
dynalignmiscloop.dat
fam_hmm_pars.dat
helix.dat
helixdr.dat
hexaloop.dat
hexaloop.dh
int11.dat
int11.dh
int21.dat
int21.dh
int22-exp.dh
int22.dat
int22.dh
loop.dat
loop.dh
m6A.coaxial.dg
m6A.coaxstack.dg
m6A.dangle.dg
m6A.hexaloop.dg
m6A.int11.dg
m6A.int21.dg
m6A.int22.dg
m6A.loop.dg
m6A.miscloop.dg
m6A.specification.dat
m6A.stack.dg
m6A.tloop.dg
m6A.triloop.dg
m6A.tstack.dg
m6A.tstackcoax.dg
m6A.tstackh.dg
m6A.tstacki.dg
m6A.tstacki1n.dg
m6A.tstacki23.dg
m6A.tstackm.dg
miscloop.dat
miscloop.dh
new_training_z_ave.scale.model
new_training_z_std.scale.model
pseudconst.dat
rna.coaxial.dg
rna.coaxial.dh
rna.coaxstack.dg
rna.coaxstack.dh
rna.cov.dg
rna.cov.dh
rna.dangle.dg
rna.dangle.dh
rna.dynalignmiscloop.dg
rna.hexaloop.dg
rna.hexaloop.dh
rna.int11.dg
rna.int11.dh
rna.int21.dg
rna.int21.dh
rna.int22.dg
rna.int22.dh
rna.loop.dg
rna.loop.dh
rna.miscloop.dg
rna.miscloop.dh
rna.param_map.dg
rna.specification.dat
rna.stack.dg
rna.stack.dh
rna.tloop.dg
rna.tloop.dh
rna.triloop.dg
rna.triloop.dh
rna.tstack.dg
rna.tstack.dh
rna.tstackcoax.dg
rna.tstackcoax.dh
rna.tstackh.dg
rna.tstackh.dh
rna.tstacki.dg
rna.tstacki.dh
rna.tstacki1n.dg
rna.tstacki1n.dh
rna.tstacki23.dg
rna.tstacki23.dh
rna.tstackm.dg
rna.tstackm.dh
rsample
stack.dat
stack.dh
stack.ds
stackdr.dat
stackdr.dh
stackdr.ds
std_ensemble_defect.model
tloop.dat
tloop.dh
triloop.dat
triloop.dh
tstack.dat
tstack.dh
tstackcoax.dat
tstackcoax.dh
tstackh.dat
tstackh.dh
tstacki.dat
tstacki.dh
tstacki1n.dat
tstacki1n.dh
tstacki23.dat
tstacki23.dh
tstackm.dat
tstackm.dh
"""


def check_data_path():
    """ Confirm the DATAPATH environment variable indicates the correct
    directory. """
    # Get the value of the DATAPATH environment variable, if it exists.
    data_path = os.environ.get(DATAPATH)
    if data_path is None:
        return f"the {DATAPATH} environment variable is not set"
    # Check if the path indicated by DATAPATH exists on the file system.
    if not os.path.exists(data_path):
        return f"{DATAPATH} is {repr(data_path)}, which does not exist"
    if not os.path.isdir(data_path):
        return f"{DATAPATH} is {repr(data_path)}, which is not a directory"
    # Check if all expected files in the DATAPATH directory exist.
    extant_files = set(os.listdir(data_path))
    for expected_file in DATAPATH_FILES.strip().split():
        if expected_file not in extant_files:
            return (f"{DATAPATH} is {repr(data_path)}, which is missing "
                    f"the required file {repr(expected_file)}")
    # All checks succeeded.
    return ""


def require_data_path():
    """ Return an error message if the DATAPATH is not valid. """
    if error := check_data_path():
        # The DATAPATH is not valid: error message.
        return (
            f"RNAstructure requires an environment variable called {DATAPATH} "
            f"to point to the directory in which its thermodynamic parameters "
            f"are located, but {error}. For more information, please refer to "
            f"https://rna.urmc.rochester.edu/Text/Thermodynamics.html"
        )
    # The DATAPATH is valid: no error.
    return ""


def fold(rna: RNAProfile, *,
         fold_temp: float,
         fold_constraint: Path | None,
         fold_md: int,
         fold_mfe: bool,
         fold_max: int,
         fold_percent: float,
         out_dir: Path,
         temp_dir: Path,
         keep_temp: bool,
         force: bool,
         n_procs: int):
    """ Run the 'Fold' or 'Fold-smp' program of RNAstructure. """
    ct_file = rna.get_ct_file(out_dir)
    if need_write(ct_file, force):
        if n_procs > 1:
            # Fold with multiple threads using the Fold-smp program.
            cmd = [RNASTRUCTURE_FOLD_SMP_CMD]
            os.environ[FOLD_SMP_NUM_THREADS] = str(n_procs)
        else:
            # Fold with one thread using the Fold program.
            cmd = [RNASTRUCTURE_FOLD_CMD]
        # Temperature of folding (Kelvin).
        cmd.extend(["--temperature", fold_temp])
        if fold_constraint is not None:
            # File of constraints.
            cmd.extend(["--constraint", fold_constraint])
        if fold_md > 0:
            # Maximum distance between paired bases.
            cmd.extend(["--maxdistance", fold_md])
        if fold_mfe:
            # Predict only the minimum free energy structure.
            cmd.append("--MFE")
        else:
            # Maximum number of structures.
            cmd.extend(["--maximum", fold_max])
            # Maximum % difference between free energies of structures.
            cmd.extend(["--percent", fold_percent])
        # DMS reactivities file for the RNA.
        cmd.extend(["--DMS", dms_file := rna.to_dms(temp_dir)])
        # Temporary FASTA file for the RNA.
        cmd.append(fasta := rna.to_fasta(temp_dir))
        # Path of the temporary CT file.
        cmd.append(ct_temp := rna.get_ct_file(temp_dir))
        try:
            # Run the command.
            run_cmd(args_to_cmd(cmd))
            # Reformat the CT file title lines so that each is unique.
            retitle_ct_structures(ct_temp, ct_temp, force=True)
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
    db_file = ct_file.with_suffix(path.DB_EXT)
    cmd = [RNASTRUCTURE_CT2DOT_CMD, ct_file, number, db_file]
    run_cmd(args_to_cmd(cmd))
    return db_file


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


def parse_rnastructure_ct_title(line: str):
    """ Parse a title in a CT file from RNAstructure, in this format:

    {length}  ENERGY = {energy}  {ref}

    where {length} is the number of positions in the structure, {ref} is
    the name of the reference, and {energy} is the predicted free energy
    of folding.
    Also handle the edge case when RNAstructure predicts no base pairs
    (and thus does not write the free energy) by returning NaN.

    Parameters
    ----------
    line: str
        Line containing the title of the structure.

    Returns
    -------
    tuple[int, float, str]
        Tuple of number of positions in the structure, predicted free
        energy of folding, and name of the reference sequence.
    """
    # Parse the line assuming it contains an energy term.
    if m := re.match(r"\s*([0-9]+)\s+ENERGY = (-?[0-9.]+)\s+(\S+)", line):
        length, energy, ref = m.groups()
    else:
        # If that failed, then parse the line assuming it does not.
        if m := re.match(r"\s*([0-9]+)\s+(\S+)", line):
            length, ref = m.groups()
        else:
            # The line violated the basic length-and-title format.
            raise ValueError(f"Failed to parse CT title line: {repr(line)}")
        logger.warning("CT line contains no energy term (probably because no "
                       f"base pairs were predicted): {repr(line)}")
        energy = "nan"
    return int(length), float(energy), ref


def format_retitled_ct_line(length: int, ref: str, uniqid: int, energy: float):
    """ Format a new CT title line including unique identifiers:

    {length}    {ref} #{uniqid}: {energy}

    where {length} is the number of positions in the structure (required
    for all CT files), {ref} is the name of the reference, {uniqid} is
    the unique identifier, and {energy} is the free energy of folding.

    Parameters
    ----------
    length: int
        Number of positions in the structure.
    uniqid: int
        Unique identifier (non-negative integer).
    ref: str
        Name of the reference.
    energy: float
        Free energy of folding.

    Returns
    -------
    str
        Formatted CT title line.
    """
    return f"{length}\t{ref} #{uniqid}: {energy}\n"


def retitle_ct_structures(ct_input: Path, ct_output: Path, force: bool = False):
    """ Retitle the structures in a CT file produced by RNAstructure.

    The default titles follow this format:

    ENERGY = {energy}  {reference}

    where {reference} is the name of the reference sequence and {energy}
    is the predicted free energy of folding.

    The major problem with this format is that structures can have equal
    predicted free energies, so the titles of the structures can repeat,
    which would cause some functions (e.g. graphing ROC curves) to fail.

    This function assigns a unique integer to each structure (starting
    with 0 for the minimum free energy and continuing upwards), which
    ensures that no two structures have identical titles.

    Parameters
    ----------
    ct_input: Path
        Path of the CT file to retitle.
    ct_output: Path
        Path of the CT file to which to write the retitled information.
    force: bool = False
        Overwrite the output CT file if it already exists.
    """
    if need_write(ct_output, force):
        # Read all lines from the input file.
        lines = list()
        with open(ct_input) as f:
            uniqid = 0
            while title_line := f.readline():
                # Parse and reformat the title line.
                n, energy, ref = parse_rnastructure_ct_title(title_line)
                lines.append(format_retitled_ct_line(n, ref, uniqid, energy))
                # Add the lines that encode the structure.
                for _ in range(n):
                    lines.append(f.readline())
                uniqid += 1
        # Write the reformatted lines to the output file.
        text = "".join(lines)
        with open(ct_output, write_mode(force)) as f:
            f.write(text)


def parse_energy(line: str):
    """ Parse the predicted free energy of folding from a line in format

    {length}    {ref} #{uniqid}: {energy}

    where {length} is the number of positions in the structure (required
    for all CT files), {ref} is the name of the reference, {uniqid} is
    the unique identifier, and {energy} is the free energy of folding.

    Parameters
    ----------
    line: str
        Line from which to parse the energy.

    Returns
    -------
    float
        Free energy of folding.
    """
    _, energy = line.split(":")
    return float(energy)

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
