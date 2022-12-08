from collections import defaultdict
import itertools
from multiprocessing import Pool
import os
import re
import shlex
from typing import List, Optional, Tuple

from dreem.util.util import FASTQC_CMD, CUTADAPT_CMD, BOWTIE2_CMD, PASTE_CMD, BASEN, \
    BOWTIE2_BUILD_CMD, TEMP_DIR, OUTPUT_DIR, DEFAULT_PROCESSES, run_cmd, \
    try_remove, switch_directory, PHRED_ENCODING, SAMTOOLS_CMD



# General parameters
DEFAULT_INTERLEAVED = False
SAM_HEADER = b"@"
SAM_ALIGN_SCORE = b"AS:i:"
SAM_EXTRA_SCORE = b"XS:i:"
FASTQ_REC_LENGTH = 4

# FastQC parameters
DEFAULT_EXTRACT = False

# Cutadapt parameters
DEFAULT_MIN_BASE_QUALITY = 25
DEFAULT_ILLUMINA_ADAPTER = "AGATCGGAAGAGC"
DEFAULT_MIN_OVERLAP = 6
DEFAULT_MAX_ERROR = 0.1
DEFAULT_INDELS = True
DEFAULT_NEXTSEQ = True
DEFAULT_DISCARD_TRIMMED = False
DEFAULT_DISCARD_UNTRIMMED = False
DEFAULT_TRIM_CORES = cpus if (cpus := os.cpu_count()) else 1
DEFAULT_MIN_LENGTH = 50

# Bowtie 2 parameters
DEFAULT_LOCAL = True
DEFAULT_UNALIGNED = False
DEFAULT_DISCORDANT = False
DEFAULT_MIXED = False
DEFAULT_DOVETAIL = False
DEFAULT_CONTAIN = True
DEFAULT_FRAG_LEN_MIN = 0
DEFAULT_FRAG_LEN_MAX = 300  # maximum length of a 150 x 150 read
DEFAULT_MAP_QUAL_MIN = 30
DEFAULT_N_CEIL = "L,0,0.05"
DEFAULT_SEED = 12
DEFAULT_EXTENSIONS = 6
DEFAULT_RESEED = 1
DEFAULT_PADDING = 4
DEFAULT_ALIGN_THREADS = os.cpu_count()
DEFAULT_METRICS = 60
MATCH_BONUS = "1"
MISMATCH_PENALTY = "1,1"
N_PENALTY = "0"
REF_GAP_PENALTY = "0,1"
READ_GAP_PENALTY = "0,1"
IGNORE_QUALS = True
SAM_EXT = ".sam"
BAM_EXT = ".bam"


def get_diffs(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences were different lengths: "
                         f"'{seq1}' and '{seq2}'")
    diffs = [i for i, (x1, x2) in enumerate(zip(seq1, seq2)) if x1 != x2]
    return diffs


def get_fastq_name(fastq: str, fastq2: Optional[str] = None):
    exts = (".fq", ".fastq")
    reads = ("mate", "r")
    base = os.path.basename(fastq)
    counts = {ext: count for ext in exts if (count := base.count(ext))}
    if not counts:
        raise ValueError(f"{fastq} had no FASTQ extension")
    if sum(counts.values()) > 1:
        raise ValueError(f"{fastq} had multiple FASTQ extensions")
    name = base[:base.index(list(counts)[0])]
    if fastq2:
        name2 = get_fastq_name(fastq2)
        diffs = get_diffs(name, name2)
        if len(diffs) != 1:
            raise ValueError("FASTQ names must differ by exactly 1 character.")
        idx = diffs[0]
        if name[idx] != "1" or name2[idx] != "2":
            raise ValueError("FASTQs must be named '1' and '2', respectively.")
        name = name[:idx]
        for read in reads:
            if name.rstrip("_").lower().endswith(read):
                name = name[:name.lower().rindex(read)]
                break
        name = name.rstrip("_")
    return name


def get_fastq_dir(fastq: str, fastq2: Optional[str] = None):
    fq_dir = os.path.dirname(fastq)
    if fastq2 and os.path.dirname(fastq2) != fq_dir:
        raise ValueError("FASTQs are not in the same directory.")
    return os.path.basename(fq_dir)


def get_fastq_pairs(fq_dir: str):
    lengths = defaultdict(set)
    for file in os.listdir(fq_dir):
        lengths[len(file)].add(file)
    pairs = dict()
    for fq_files in lengths.values():
        while fq_files:
            file1 = fq_files.pop()
            diffs = defaultdict(set)
            for file2 in fq_files:
                diffs[len(get_diffs(file1, file2))].add(file2)
            try:
                file2 = diffs[1].pop()
            except KeyError:
                raise ValueError(f"Found no mate for {file1} in {fq_dir}")
            if diffs[1]:
                raise ValueError(f"Found >1 mate for {file1} in {fq_dir}")
            fq_files.pop(file2)
            pair = (os.path.join(fq_dir, file1), os.path.join(fq_dir, file2))
            pairs[get_fastq_name(file1, file2)] = pair
    return pairs


class SeqFileBase(object):
    _operation_dir = ""

    def __init__(self,
                 root_dir: str,
                 ref_file: str,
                 sample: str,
                 paired: bool,
                 encoding: int = PHRED_ENCODING) -> None:
        self._root_dir = root_dir
        self._ref_file = ref_file
        self._sample = sample
        self._paired = paired
        self._encoding = encoding
    
    @property
    def operation_dir(self):
        if not self._operation_dir:
            raise NotImplementedError
        return self._operation_dir
    
    @property
    def root_dir(self):
        return self._root_dir
    
    @property
    def ref_file(self):
        return self._ref_file
    
    @property
    def ref_prefix(self):
        return os.path.splitext(self.ref_file)[0]
    
    @property
    def ref_filename(self):
        return os.path.basename(self.ref_prefix)
    
    @property
    def sample(self):
        return self._sample

    @property
    def paired(self):
        return self._paired
    
    @property
    def encoding(self):
        return self._encoding
    
    @property
    def encoding_arg(self):
        return f"--phred{self.encoding}"
    
    def _get_dir(self, dest: str):
        return os.path.join(self.root_dir, dest,
                            self.operation_dir, self.sample)
    
    @property
    def temp_dir(self):
        return self._get_dir(TEMP_DIR)
    
    @property
    def output_dir(self):
        return self._get_dir(OUTPUT_DIR)
    
    def make_temp_dir(self):
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def make_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(*args, **kwargs):
        raise NotImplementedError


class FastqBase(SeqFileBase):
    def __init__(self,
                 root_dir: str,
                 ref_file: str,
                 sample: str,
                 fastq: str,
                 paired: bool,
                 encoding: int = PHRED_ENCODING) -> None:
        super().__init__(root_dir, ref_file, sample, paired, encoding=encoding)
        self._fastq = fastq
    
    @property
    def fastq(self):
        return self._fastq
    
    @property
    def temp_output(self):
        return switch_directory(self.fastq, self.temp_dir)
    
    def qc(self, extract: bool = DEFAULT_EXTRACT):
        cmd = [FASTQC_CMD]
        if extract:
            cmd.append("--extract")
        cmd.append(self.fastq)
        run_cmd(cmd)


class FastqInterleaver(FastqBase):
    _operation_dir = "alignment/interleave"

    def __init__(self,
                 root_dir: str,
                 ref_file: str,
                 sample: str,
                 fastq1: str,
                 fastq2: str,
                 encoding: int = PHRED_ENCODING) -> None:
        super().__init__(root_dir, ref_file, sample, fastq1, paired=True,
                         encoding=encoding)
        self._fastq2 = fastq2
    
    @property
    def fastq1(self):
        return self._fastq
    
    @property
    def fastq2(self):
        return self._fastq2
    
    @property
    def fastq(self):
        return f"{self.sample}.fq"
    
    @staticmethod
    def _read_record(fq_file):
        return b"".join(itertools.islice(fq_file, FASTQ_REC_LENGTH))
    
    @classmethod
    def _read_records(cls, fq_file):
        while record := cls._read_record(fq_file):
            yield record
    
    @classmethod
    def _interleave(cls, fastq1: str, fastq2: str, output: str):
        with (open(fastq1, "rb") as fq1, open(fastq2, "rb") as fq2,
              open(output, "wb") as fqo):
            for record in itertools.chain.from_iterable(zip(
                    cls._read_records(fq1), cls._read_records(fq2))):
                fqo.write(record)
    
    def run(self):
        self.make_temp_dir()
        self._interleave(self.fastq1, self.fastq2, self.temp_output)
        return self.temp_output


class FastqMasker(FastqBase):
    _operation_dir = "alignment/mask"

    def _mask(self, min_qual: int):
        min_code = min_qual + self.encoding
        NL = b"\n"[0]
        BN = BASEN[0]
        with open(self.fastq, "rb") as fqi, open(self.temp_output, "wb") as fqo:
            for seq_header in fqi:
                masked = bytearray(seq_header)
                seq = fqi.readline()
                qual_header = fqi.readline()
                quals = fqi.readline()
                if len(seq) != len(quals):
                    raise ValueError("seq and qual have different lengths")
                masked.extend(base if qual >= min_code or base == NL else BN
                              for base, qual in zip(seq, quals))
                masked.extend(qual_header)
                masked.extend(quals)
                fqo.write(masked)
    
    def run(self, min_qual: int = DEFAULT_MIN_BASE_QUALITY):
        self.make_temp_dir()
        self._mask(min_qual)
        return self.temp_output


class FastqTrimmer(FastqBase):
    _operation_dir = "alignment/trim"

    def _cutadapt(self,
                  qual1=DEFAULT_MIN_BASE_QUALITY,
                  qual2=None,
                  adapters15: Tuple[str] = (),
                  adapters13: Tuple[str] = (DEFAULT_ILLUMINA_ADAPTER,),
                  adapters25: Tuple[str] = (),
                  adapters23: Tuple[str] = (DEFAULT_ILLUMINA_ADAPTER,),
                  min_overlap=DEFAULT_MIN_OVERLAP,
                  max_error=DEFAULT_MAX_ERROR,
                  indels=DEFAULT_INDELS,
                  nextseq=DEFAULT_NEXTSEQ,
                  discard_trimmed=DEFAULT_DISCARD_TRIMMED,
                  discard_untrimmed=DEFAULT_DISCARD_UNTRIMMED,
                  min_length=DEFAULT_MIN_LENGTH,
                  cores=DEFAULT_TRIM_CORES):
        cmd = [CUTADAPT_CMD]
        if cores >= 0:
            cmd.extend(["--cores", str(cores)])
        if nextseq:
            nextseq_qual = qual1 if qual1 else DEFAULT_MIN_BASE_QUALITY
            cmd.extend(["--nextseq-trim", str(nextseq_qual)])
        else:
            if qual1 is not None:
                cmd.extend(["-q", str(qual1)])
            if qual2 is not None:
                self.fastq2
                cmd.extend(["-Q", str(qual2)])
        adapters = {"g": adapters15, "a": adapters13,
                    "G": adapters25, "A": adapters23}
        for arg, adapter in adapters.items():
            if adapter and (arg.islower() or self.paired):
                if isinstance(adapter, str):
                    adapter = (adapter,)
                if not isinstance(adapter, tuple):
                    raise ValueError("adapters must be str or tuple")
                for adapt in adapter:
                    cmd.extend([f"-{arg}", adapt])
        if min_overlap >= 0:
            cmd.extend(["-O", str(min_overlap)])
        if max_error >= 0:
            cmd.extend(["-e", str(max_error)])
        if not indels:
            cmd.append("--no-indels")
        if discard_trimmed:
            cmd.append("--discard-trimmed")
        if discard_untrimmed:
            cmd.append("--discard-untrimmed")
        if min_length:
            cmd.extend(["-m", str(min_length)])
        cmd.append("--interleaved")
        cmd.extend(["-o", self.temp_output])
        cmd.append(self.fastq)
        run_cmd(cmd)
    
    def run(self, **kwargs):
        self.make_temp_dir()
        self._cutadapt(**kwargs)
        return self.temp_output


class FastqAligner(FastqBase):
    _operation_dir = "alignment/bowtie2"

    @property
    def temp_output(self):
        return os.path.join(self.temp_dir, f"{self.ref_filename}.sam")

    def _bowtie2_build(self):
        """
        Build an index of a reference genome using Bowtie 2.
        :param ref: (str) path to the reference genome FASTA file
        :return: None
        """
        cmd = [BOWTIE2_BUILD_CMD, self.ref_file, self.ref_prefix]
        run_cmd(cmd)
    
    def _bowtie2(self,
                 local=DEFAULT_LOCAL,
                 unaligned=DEFAULT_UNALIGNED,
                 discordant=DEFAULT_DISCORDANT,
                 mixed=DEFAULT_MIXED,
                 dovetail=DEFAULT_DOVETAIL,
                 contain=DEFAULT_CONTAIN,
                 frag_len_min=DEFAULT_FRAG_LEN_MIN,
                 frag_len_max=DEFAULT_FRAG_LEN_MAX,
                 map_qual_min=DEFAULT_MAP_QUAL_MIN,
                 n_ceil=DEFAULT_N_CEIL,
                 seed=DEFAULT_SEED,
                 extensions=DEFAULT_EXTENSIONS,
                 reseed=DEFAULT_RESEED,
                 padding=DEFAULT_PADDING,
                 threads=DEFAULT_ALIGN_THREADS,
                 metrics=DEFAULT_METRICS):
        cmd = [BOWTIE2_CMD]
        fastq_flag = "--interleaved" if self.paired else "-U"
        cmd.extend([fastq_flag, self.fastq])
        cmd.extend(["-x", self.ref_prefix])
        cmd.extend(["-S", self.temp_output])
        cmd.append(self.encoding_arg)
        cmd.append("--xeq")
        cmd.extend(["--ma", MATCH_BONUS])
        cmd.extend(["--mp", MISMATCH_PENALTY])
        cmd.extend(["--np", N_PENALTY])
        cmd.extend(["--rfg", REF_GAP_PENALTY])
        cmd.extend(["--rdg", READ_GAP_PENALTY])
        if local:
            cmd.append("--local")
        if not unaligned:
            cmd.append("--no-unal")
        if not discordant:
            cmd.append("--no-discordant")
        if not mixed:
            cmd.append("--no-mixed")
        if dovetail:
            cmd.append("--dovetail")
        if not contain:
            cmd.append("--no-contain")
        if frag_len_min:
            cmd.extend(["-I", str(frag_len_min)])
        if frag_len_max:
            cmd.extend(["-X", str(frag_len_max)])
        #if map_qual_min:
        #    cmd.extend(["--score-min", f"C,{map_qual_min}"])
        if n_ceil:
            cmd.extend(["--n-ceil", n_ceil])
        if seed:
            cmd.extend(["-L", str(seed)])
        if extensions:
            cmd.extend(["-D", str(extensions)])
        if reseed:
            cmd.extend(["-R", str(reseed)])
        if padding:
            cmd.extend(["--dpad", str(padding)])
        if threads:
            cmd.extend(["-p", str(threads)])
        if metrics:
            cmd.extend(["--met-stderr", "--met", str(metrics)])
        if IGNORE_QUALS:
            cmd.append("--ignore-quals")
        run_cmd(cmd)
    
    def run(self, **kwargs):
        self.make_temp_dir()
        self._bowtie2_build()
        self._bowtie2(**kwargs)


class AlignmentCleaner(SeqFileBase):
    _operation_dir = "alignment/cleaned"

    def remove_equal_mappers(self):
        print("\n\nRemoving reads mapping equally to multiple locations")
        pattern_a = re.compile(SAM_ALIGN_SCORE + rb"(\d+)")
        pattern_x = re.compile(SAM_EXTRA_SCORE + rb"(\d+)")

        def get_score(line, ptn):
            return (float(match.groups()[0])
                    if (match := ptn.search(line)) else None)

        def is_best_alignment(line):
            return ((score_x := get_score(line, pattern_x)) is None
                    or score_x < get_score(line, pattern_a))

        kept = 0
        removed = 0
        with open(self.sam_input, "rb") as sami, open(self.sam_temp, "wb") as samo:
            # Copy the header from the input to the output SAM file.
            while (line := sami.readline()).startswith(SAM_HEADER):
                samo.write(line)
            if self.paired:
                for line2 in sami:
                    if is_best_alignment(line) or is_best_alignment(line2):
                        samo.write(line)
                        samo.write(line2)
                        kept += 1
                    else:
                        removed += 1
                    line = sami.readline()
            else:
                while line:
                    if is_best_alignment(line):
                        samo.write(line)
                        kept += 1
                    else:
                        removed += 1
                    line = sami.readline()
        total = kept + removed
        items = "Pairs" if self.paired else "Reads"
        print(f"\n{items} processed: {total}")
        print(f"{items} kept: {kept} ({round(100 * kept / total, 2)}%)")
        print(f"{items} lost: {removed} ({round(100 * removed / total, 2)}%)")
        return kept, removed


class AlignmentFinisher(SeqFileBase):
    _operation_dir = "align"

    def sort(self):
        cmd = [SAMTOOLS_CMD, "sort", "-o", self.bam_out, self.sam_input]
        run_cmd(cmd)
    
    def index(self):
        cmd = [SAMTOOLS_CMD, "index", "-b", self.bam_out]
        run_cmd(cmd)


def run(output_dir, ref, fastq1, fastq2=None, **kwargs):
    # Check the file extensions.
    primer1 = "CAGCACTCAGAGCTAATACGACTCACTATA"
    primer1rc = "TATAGTGAGTCGTATTAGCTCTGAGTGCTG"
    primer2 = "TGAAGAGCTGGAACGCTTCACTGA"
    primer2rc = "TCAGTGAAGCGTTCCAGCTCTTCA"
    adapters5 = (primer1, primer2rc)
    adapters3 = (primer2, primer1rc)
    primer_trimmed_reads = os.path.join(output_dir,
        f"__{sample}_primer_trimmed.fastq")
    trim(adapter_trimmed_reads, primer_trimmed_reads, interleaved=paired,
         adapters15=adapters5, adapters13=adapters3,
         adapters25=adapters5, adapters23=adapters3,
         min_overlap=6, min_length=10, discard_trimmed=True)
    try_remove(adapter_trimmed_reads)
    # Mask low-quality bases remaining in the trimmed file.
    masked_trimmed_reads = os.path.join(output_dir,
        f"__{sample}_masked_trimmed.fastq")
    mask_low_qual(primer_trimmed_reads, masked_trimmed_reads)
    try_remove(primer_trimmed_reads)
    # Quality check the trimmed FASTQ file.
    #fastqc(output_dir, masked_trimmed_reads)
    # Index the reference.
    prefix = index(ref)
    # Align the reads to the reference.
    sam_align = os.path.join(output_dir, f"__{sample}_align.sam")
    bowtie(sam_align, prefix, masked_trimmed_reads, interleaved=True)
    try_remove(masked_trimmed_reads)
    # Delete reads aligning equally well to multiple locations.
    sam_pruned = os.path.join(output_dir, f"__{sample}_pruned.sam")
    remove_equal_mappers(sam_align, sam_pruned)
    # Delete the non-pruned SAM file.
    try_remove(sam_align)
    # Sort the SAM file while converting to BAM format
    bam_file = os.path.join(output_dir, f"{sample}.bam")
    sort_sam(sam_pruned, bam_file)
    # Delete the SAM file.
    try_remove(sam_pruned)
    # Index the BAM file.
    index_bam(bam_file)
