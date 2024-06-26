from logging import getLogger
from pathlib import Path

from click import command

from .write import align_samples
from .fqops import FastqUnit
from ..core.arg import (CMD_ALIGN,
                        docdef,
                        arg_fasta,
                        opt_fastqz,
                        opt_fastqy,
                        opt_fastqx,
                        opt_dmfastqz,
                        opt_dmfastqy,
                        opt_dmfastqx,
                        opt_phred_enc,
                        opt_out_dir,
                        opt_temp_dir,
                        opt_force,
                        opt_keep_temp,
                        opt_parallel,
                        opt_max_procs,
                        opt_fastqc,
                        opt_qc_extract,
                        opt_cutadapt,
                        opt_cut_a1,
                        opt_cut_g1,
                        opt_cut_a2,
                        opt_cut_g2,
                        opt_cut_o,
                        opt_cut_e,
                        opt_cut_q1,
                        opt_cut_q2,
                        opt_cut_m,
                        opt_cut_indels,
                        opt_cut_discard_trimmed,
                        opt_cut_discard_untrimmed,
                        opt_cut_nextseq,
                        opt_bt2_local,
                        opt_bt2_discordant,
                        opt_bt2_mixed,
                        opt_bt2_dovetail,
                        opt_bt2_contain,
                        opt_bt2_i,
                        opt_bt2_x,
                        opt_bt2_score_min_loc,
                        opt_bt2_score_min_e2e,
                        opt_bt2_s,
                        opt_bt2_l,
                        opt_bt2_d,
                        opt_bt2_r,
                        opt_bt2_gbar,
                        opt_bt2_dpad,
                        opt_bt2_orient,
                        opt_bt2_un,
                        opt_min_mapq,
                        opt_min_reads,
                        opt_cram)
from ..core.extern import (BOWTIE2_CMD,
                           BOWTIE2_BUILD_CMD,
                           CUTADAPT_CMD,
                           FASTQC_CMD,
                           SAMTOOLS_CMD,
                           require_dependency)
from ..core.parallel import lock_temp_dir

logger = getLogger(__name__)


@lock_temp_dir
@docdef.auto()
def run(*,
        # Inputs
        fasta: str,
        fastqz: tuple[str, ...],
        fastqy: tuple[str, ...],
        fastqx: tuple[str, ...],
        dmfastqz: tuple[str, ...],
        dmfastqy: tuple[str, ...],
        dmfastqx: tuple[str, ...],
        phred_enc: int,
        # Outputs
        out_dir: str,
        temp_dir: str,
        keep_temp: bool,
        force: bool,
        # Parallelization
        max_procs: int,
        parallel: bool,
        # FASTQC
        fastqc: bool,
        qc_extract: bool,
        # Cutadapt
        cut: bool,
        cut_q1: int,
        cut_q2: int,
        cut_g1: tuple[str, ...],
        cut_a1: tuple[str, ...],
        cut_g2: tuple[str, ...],
        cut_a2: tuple[str, ...],
        cut_o: int,
        cut_e: float,
        cut_indels: bool,
        cut_nextseq: bool,
        cut_discard_trimmed: bool,
        cut_discard_untrimmed: bool,
        cut_m: int,
        # Bowtie2
        bt2_local: bool,
        bt2_discordant: bool,
        bt2_mixed: bool,
        bt2_dovetail: bool,
        bt2_contain: bool,
        bt2_score_min_e2e: str,
        bt2_score_min_loc: str,
        bt2_i: int,
        bt2_x: int,
        bt2_gbar: int,
        bt2_l: int,
        bt2_s: str,
        bt2_d: int,
        bt2_r: int,
        bt2_dpad: int,
        bt2_orient: str,
        bt2_un: bool,
        # Samtools
        min_mapq: int,
        min_reads: int,
        cram: bool) -> list[Path]:
    """ Trim FASTQ files and align them to reference sequences. """
    # Check for external dependencies.
    if fastqc and (error := require_dependency(FASTQC_CMD, __name__)):
        logger.critical(error)
        return list()
    if cut and (error := require_dependency(CUTADAPT_CMD, __name__)):
        logger.critical(error)
        return list()
    if error := require_dependency(BOWTIE2_CMD, __name__):
        logger.critical(error)
        return list()
    if error := require_dependency(BOWTIE2_BUILD_CMD, __name__):
        logger.critical(error)
        return list()
    if error := require_dependency(SAMTOOLS_CMD, __name__):
        logger.critical(error)
        return list()
    # FASTQ files of read sequences may come from up to seven different
    # sources (i.e. each argument beginning with "fq_unit"). This step
    # collects all of them into one list (fq_units) and also bundles
    # together pairs of FASTQ files containing mate 1 and mate 2 reads.
    fq_units = list(FastqUnit.from_paths(fastqz=list(map(Path, fastqz)),
                                         fastqy=list(map(Path, fastqy)),
                                         fastqx=list(map(Path, fastqx)),
                                         dmfastqz=list(map(Path, dmfastqz)),
                                         dmfastqy=list(map(Path, dmfastqy)),
                                         dmfastqx=list(map(Path, dmfastqx)),
                                         phred_enc=phred_enc))

    # Generate and return a BAM file for every FASTQ-reference pair.
    return align_samples(fq_units=fq_units,
                         fasta=Path(fasta),
                         out_dir=Path(out_dir),
                         temp_dir=Path(temp_dir),
                         keep_temp=keep_temp,
                         force=force,
                         max_procs=max_procs,
                         parallel=parallel,
                         fastqc=fastqc,
                         qc_extract=qc_extract,
                         cut=cut,
                         cut_q1=cut_q1,
                         cut_q2=cut_q2,
                         cut_g1=cut_g1,
                         cut_a1=cut_a1,
                         cut_g2=cut_g2,
                         cut_a2=cut_a2,
                         cut_o=cut_o,
                         cut_e=cut_e,
                         cut_indels=cut_indels,
                         cut_nextseq=cut_nextseq,
                         cut_discard_trimmed=cut_discard_trimmed,
                         cut_discard_untrimmed=cut_discard_untrimmed,
                         cut_m=cut_m,
                         bt2_local=bt2_local,
                         bt2_discordant=bt2_discordant,
                         bt2_mixed=bt2_mixed,
                         bt2_dovetail=bt2_dovetail,
                         bt2_contain=bt2_contain,
                         bt2_un=bt2_un,
                         bt2_score_min_e2e=bt2_score_min_e2e,
                         bt2_score_min_loc=bt2_score_min_loc,
                         bt2_i=bt2_i,
                         bt2_x=bt2_x,
                         bt2_gbar=bt2_gbar,
                         bt2_l=bt2_l,
                         bt2_s=bt2_s,
                         bt2_d=bt2_d,
                         bt2_r=bt2_r,
                         bt2_dpad=bt2_dpad,
                         bt2_orient=bt2_orient,
                         min_mapq=min_mapq,
                         min_reads=min_reads,
                         cram=cram)


# Parameters for command line interface
params = [
    # Inputs
    arg_fasta,
    opt_fastqx,
    opt_fastqy,
    opt_fastqz,
    opt_dmfastqx,
    opt_dmfastqy,
    opt_dmfastqz,
    opt_phred_enc,
    # Outputs
    opt_out_dir,
    opt_temp_dir,
    opt_force,
    opt_keep_temp,
    # Parallelization
    opt_parallel,
    opt_max_procs,
    # FASTQC
    opt_fastqc,
    opt_qc_extract,
    # Cutadapt
    opt_cutadapt,
    opt_cut_a1,
    opt_cut_g1,
    opt_cut_a2,
    opt_cut_g2,
    opt_cut_o,
    opt_cut_e,
    opt_cut_q1,
    opt_cut_q2,
    opt_cut_m,
    opt_cut_indels,
    opt_cut_discard_trimmed,
    opt_cut_discard_untrimmed,
    opt_cut_nextseq,
    # Bowtie2
    opt_bt2_local,
    opt_bt2_discordant,
    opt_bt2_mixed,
    opt_bt2_dovetail,
    opt_bt2_contain,
    opt_bt2_i,
    opt_bt2_x,
    opt_bt2_score_min_e2e,
    opt_bt2_score_min_loc,
    opt_bt2_s,
    opt_bt2_l,
    opt_bt2_gbar,
    opt_bt2_d,
    opt_bt2_r,
    opt_bt2_dpad,
    opt_bt2_orient,
    opt_bt2_un,
    # Samtools
    opt_min_mapq,
    opt_min_reads,
    opt_cram,
]


@command(CMD_ALIGN, params=params)
def cli(*args, **kwargs):
    """ Trim FASTQ files and align them to reference sequences. """
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
