#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:45:25 2024

@author: justin
"""

from seismicrna.core.rna.profile import RNAProfile
from pathlib import Path
from seismicrna.core.seq.xna import XNA, RNA
from seismicrna.core.path import STR_CHARS
from seismicrna.core.write import write_mode
import re
from pathlib import Path
from typing import Iterable
from seismicrna.core.seq import format_fasta_record

# FASTA name line format.
FASTA_NAME_MARK = ">"
FASTA_NAME_CHARS = STR_CHARS
FASTA_NAME_REGEX = re.compile(f"^{FASTA_NAME_MARK}([{FASTA_NAME_CHARS}]*)")

    
def write_fasta(fasta: Path,
                refs: Iterable[tuple[str, XNA]],
                edited_positions: Iterable[int],
                wrap: int = 0,
                force: bool = False):
    """ Write an iterable of reference names and DNA sequences to a
    FASTA file. """
    # Record the names of all the references.
    names = set()
    with open(fasta, write_mode(force)) as f:
        for name, seq in refs:
            seq_list = list(seq)
            seq_list_edited = list()
            for idx, base in enumerate(seq_list):
                if idx not in edited_positions:
                    seq_list_edited.append(base) 
                else:
                    seq_list_edited.append("G")
            edited_seq = "".join(seq_list_edited)
            seq = RNA(edited_seq)
            try:
                # Confirm that the name is not blank.
                if not name:
                    raise ValueError("Got blank reference name")
                # Confirm there are no illegal characters in the name.
                if illegal := set(name) - set(FASTA_NAME_CHARS):
                    raise ValueError(f"Reference name '{name}' has illegal "
                                     f"characters: {illegal}")
                # If there are two or more references with the same name,
                # then the sequence of only the first is used.
                if name in names:
                    raise ValueError(f"Duplicate reference name: '{name}'")
                f.write(format_fasta_record(name, seq, wrap))
                names.add(name)
            except Exception:
                raise
                
def to_fasta_deconvolve(self, top: Path, edited_positions: Iterable[int]):
    """ Write the RNA sequence to a FASTA file.

    Parameters
    ----------
    top: pathlib.Path
        Top-level directory.

    Returns
    -------
    pathlib.Path
        File into which the RNA sequence was written.
    """
    fasta = self.get_fasta(top)
    write_fasta(fasta, [self.seq_record], edited_positions)
    return fasta