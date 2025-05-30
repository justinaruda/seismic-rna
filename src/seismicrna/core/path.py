from __future__ import annotations

import os
import pathlib
import re
import shutil
from collections import Counter
from functools import cache, cached_property, partial, wraps
from itertools import chain, product
from string import ascii_letters, digits, printable
from tempfile import mkdtemp
from typing import Any, Callable, Iterable, Sequence

from .logs import logger

# Constants ############################################################

INP_TEST_DIR = pathlib.Path.cwd().joinpath("test-inp")
OUT_TEST_DIR = pathlib.Path.cwd().joinpath("test-out")
TMP_TEST_DIR = pathlib.Path.cwd().joinpath("test-tmp")

# Valid/invalid characters in fields

PATH_CHARS = printable
ALPHANUM_CHARS = ascii_letters + digits
STR_CHARS = ALPHANUM_CHARS + "_.=+-"
STR_CHARS_SET = frozenset(STR_CHARS)
INT_CHARS = digits
PATH_PATTERN = f"([{PATH_CHARS}]+)"
STR_PATTERN = f"([{STR_CHARS}]+)"
INT_PATTERN = f"([{INT_CHARS}]+)"
RE_PATTERNS = {str: STR_PATTERN, int: INT_PATTERN, pathlib.Path: PATH_PATTERN}


# Names of steps
ALIGN_STEP = "align"
RELATE_STEP = "relate"
NAMES_STEP = "names"
MASK_STEP = "mask"
CLUSTER_STEP = "cluster"
DECONVOLVE_STEP = "deconvolve"
LIST_STEP = "list"
FOLD_STEP = "fold"
GRAPH_STEP = "graph"

# Directories for simulation

SIM_REFS_DIR = "refs"
SIM_PARAM_DIR = "params"
SIM_SAMPLES_DIR = "samples"

# Directories for stages

STAGE_ALIGN_INDEX = "index"
STAGE_ALIGN_INDEX_DEMULT = "index-demult"
STAGE_ALIGN_TRIM = "trim"
STAGE_ALIGN_MAP = "map"
STAGE_ALIGN_SORT = "sort"

STAGE_REL_SAMS = "sams"

STAGES = (STAGE_ALIGN_INDEX,
          STAGE_ALIGN_INDEX_DEMULT,
          STAGE_ALIGN_TRIM,
          STAGE_ALIGN_MAP,
          STAGE_ALIGN_SORT,
          STAGE_REL_SAMS)

# Cluster information
CLUST_PARAM_PIS = "pis"
CLUST_PARAM_MUS = "mus"
CLUST_PARAMS = (CLUST_PARAM_PIS,
                CLUST_PARAM_MUS)
CLUST_PARAMS_DIR = "parameters"
CLUST_STATS_DIR = "statistics"
CLUST_COUNTS_DIR = "read-counts"

TABLES = (RELATE_STEP, MASK_STEP, CLUSTER_STEP, DECONVOLVE_STEP)

# File extensions

GZIP_EXT = ".gz"
TXT_EXT = ".txt"
CSV_EXT = ".csv"
CSVZIP_EXT = ".csv.gz"
CSV_EXTS = CSV_EXT, CSVZIP_EXT
BROTLI_PICKLE_EXT = ".brickle"
JSON_EXT = ".json"
FASTA_EXTS = ".fa", ".fna", ".fasta"
BOWTIE2_INDEX_EXTS = (".1.bt2",
                      ".2.bt2",
                      ".3.bt2",
                      ".4.bt2",
                      ".rev.1.bt2",
                      ".rev.2.bt2")
FQ_EXTS = (".fq.gz",
           ".fastq.gz",
           ".fq",
           ".fastq",
           "_001.fq.gz",
           "_001.fastq.gz",
           "_001.fq",
           "_001.fastq")
SVG_EXT = ".svg"
FQ_PAIRED_EXTS_TEMPLATES = "_R{}{}", "_{}{}", "_mate{}{}", "_{}_sequence{}"
FQ1_EXTS = tuple(template.format(1, ext) for template, ext in
                 product(FQ_PAIRED_EXTS_TEMPLATES, FQ_EXTS))
FQ2_EXTS = tuple(template.format(2, ext) for template, ext in
                 product(FQ_PAIRED_EXTS_TEMPLATES, FQ_EXTS))
SAM_EXT = ".sam"
BAM_EXT = ".bam"
CRAM_EXT = ".cram"
XAM_EXTS = SAM_EXT, BAM_EXT, CRAM_EXT
FAI_EXT = ".fai"
CT_EXT = ".ct"
DB_EXT = ".db"
DBN_EXT = ".dbn"
DOT_EXT = ".dot"
DOT_EXTS = DB_EXT, DBN_EXT, DOT_EXT
DMS_EXT = ".dms"
KTS_EXT = ".kts"
HTML_EXT = ".html"
SVG_EXT = ".svg"
KTS_EXT = ".kts"
PDF_EXT = ".pdf"
PNG_EXT = ".png"
GRAPH_EXTS = CSV_EXT, HTML_EXT, SVG_EXT, PDF_EXT, PNG_EXT
PARAM_MUTS_EXT = f".muts{CSV_EXT}"
PARAM_ENDS_EXT = f".ends{CSV_EXT}"
PARAM_CLUSTS_EXT = f".clusts{CSV_EXT}"


# Path Exceptions ######################################################

class PathError(Exception):
    """ Any error involving a path. """


class PathTypeError(PathError, TypeError):
    """ Use of the wrong type of path or segment. """


class PathValueError(PathError, ValueError):
    """ Invalid value of a path segment field. """


class WrongFileExtensionError(PathValueError):
    """ A file has the wrong extension. """


# Path Functions #######################################################

def fill_whitespace(path: str | Path, fill: str = "_"):
    """ Replace all whitespace in `path` with `fill`. """
    return path.__class__(fill.join(str(path).split()))


def sanitize(path: str | pathlib.Path, strict: bool = False):
    """ Sanitize a path-like object by ensuring it is an absolute path,
    eliminating symbolic links and redundant path separators/references,
    and returning a Path object.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to sanitize.
    strict: bool = False
        Require the path to exist and contain no symbolic link loops.

    Returns
    -------
    pathlib.Path
        Absolute, normalized, symlink-free path.
    """
    return pathlib.Path(path).resolve(strict=strict)


@cache
def get_seismicrna_source_dir():
    """ SEISMIC-RNA source directory, named seismicrna, containing
    __init__.py and the top-level modules and subpackages. """
    seismicrna_src_dir = sanitize(__file__, strict=True).parent.parent
    try:
        from seismicrna import __file__ as seismicrna_file
    except ImportError:
        seismicrna_file = None
    if seismicrna_file:
        seismicrna_parent = sanitize(seismicrna_file).parent
        if seismicrna_parent != seismicrna_src_dir:
            raise PathValueError("Inconsistent source directory: "
                                 f"{seismicrna_src_dir} ≠ {seismicrna_parent}")
    else:
        logger.warning("seismicrna is not installed: skipped verifying path")
    name = "seismicrna"
    if seismicrna_src_dir.name != name:
        raise PathValueError(f"Source directory {seismicrna_src_dir} "
                             f"is not named {repr(name)}")
    return seismicrna_src_dir


@cache
def get_seismicrna_project_dir():
    """ SEISMIC-RNA project directory, named seismic-rna, containing
    src, pyproject.toml, and all other project files. Will exist if the
    entire SEISMIC-RNA project has been downloaded, e.g. from GitHub,
    but not if SEISMIC-RNA was only installed using pip or conda. """
    seismicrna_prj_dir = get_seismicrna_source_dir().parent.parent
    name = "seismic-rna"
    if seismicrna_prj_dir.name != name:
        # It is fine if the project directory does not exist because
        # installing SEISMIC-RNA using pip or conda installs only the
        # source directory, but not the project directory.
        return None
    return seismicrna_prj_dir


# Path Fields ##########################################################

# Field validation functions

def validate_str(txt: str):
    if not isinstance(txt, str):
        raise PathTypeError(
            f"Expected {str.__name__}, but got {type(txt).__name__}"
        )
    if not txt:
        raise PathValueError(f"Empty string: {repr(txt)}")
    if illegal := "".join(sorted(set(txt) - STR_CHARS_SET)):
        raise PathValueError(f"{repr(txt)} has illegal characters: {illegal}")


def validate_top(top: pathlib.Path):
    if not isinstance(top, pathlib.Path):
        raise PathTypeError(
            f"Expected {Path.__name__}, but got {type(top).__name__}"
        )
    if not top.parent.is_dir():
        raise PathValueError(f"Not a directory: {top.parent}")
    if top.is_file():
        raise PathValueError(f"File exists: {top}")


def validate_int(num: int):
    if not isinstance(num, int) or num < 0:
        raise PathValueError(f"Expected an integer ≥ 0, but got {num}")


VALIDATE = {int: validate_int,
            str: validate_str,
            pathlib.Path: validate_top}


# Field class

class Field(object):
    def __init__(self,
                 dtype: type[str | int | pathlib.Path],
                 options: Iterable = (),
                 is_ext: bool = False):
        self.dtype = dtype
        self.options = list(options)
        if not all(isinstance(option, self.dtype) for option in self.options):
            raise PathTypeError("All options of a field must be of its type")
        self.is_ext = is_ext
        if self.is_ext:
            if self.dtype is not str:
                raise PathTypeError("Extension field must be type 'str', "
                                    f"but got type {repr(self.dtype.__name__)}")
            if not self.options:
                raise PathValueError("Extension field must have options")

    def validate(self, val: Any):
        if not isinstance(val, self.dtype):
            raise PathTypeError(f"Expected {repr(self.dtype.__name__)}, but "
                                f"got {repr(val)} ({repr(type(val).__name__)})")
        if self.options and val not in self.options:
            raise PathValueError(
                f"Invalid option {repr(val)}; expected one of {self.options}"
            )
        VALIDATE[self.dtype](val)

    def build(self, val: Any):
        """ Validate a value and return it as a string. """
        self.validate(val)
        return str(val)

    def parse(self, text: str) -> Any:
        """ Parse a value from a string, validate it, and return it. """
        try:
            val = self.dtype(text)
        except Exception as error:
            raise PathValueError(
                f"Failed to interpret {repr(text)} as type "
                f"{repr(self.dtype.__name__)}: {error}"
            ) from None
        self.validate(val)
        return val

    @cached_property
    def as_str(self):
        # Define the string as a cached property to speed up str(self).
        return f"{type(self).__name__} <{self.dtype.__name__}>"

    def __str__(self):
        return self.as_str


# Fields
TopField = Field(pathlib.Path)
NameField = Field(str)
CmdField = Field(str, [ALIGN_STEP,
                       RELATE_STEP,
                       MASK_STEP,
                       CLUSTER_STEP,
                       DECONVOLVE_STEP,
                       LIST_STEP,
                       FOLD_STEP,
                       GRAPH_STEP])
StageField = Field(str, STAGES)
IntField = Field(int)
ClustRunResultsField = Field(str, CLUST_PARAMS)
PosTableField = Field(str, TABLES)
ReadTableField = Field(str, TABLES)
AbundanceField = Field(str, [CLUSTER_STEP, DECONVOLVE_STEP])

# File extensions
TextExt = Field(str, [TXT_EXT], is_ext=True)
ReportExt = Field(str, [JSON_EXT], is_ext=True)
RefseqFileExt = Field(str, [BROTLI_PICKLE_EXT], is_ext=True)
BatchExt = Field(str, [BROTLI_PICKLE_EXT], is_ext=True)
ClustTabExt = Field(str, CSV_EXTS, is_ext=True)
PosTableExt = Field(str, [CSV_EXT], is_ext=True)
ReadTableExt = Field(str, [CSVZIP_EXT], is_ext=True)
AbundanceExt = Field(str, [CSV_EXT], is_ext=True)
FastaExt = Field(str, FASTA_EXTS, is_ext=True)
FastaIndexExt = Field(str, BOWTIE2_INDEX_EXTS, is_ext=True)
FastqExt = Field(str, FQ_EXTS, is_ext=True)
Fastq1Ext = Field(str, FQ1_EXTS, is_ext=True)
Fastq2Ext = Field(str, FQ2_EXTS, is_ext=True)
XamExt = Field(str, XAM_EXTS, is_ext=True)
ConnectTableExt = Field(str, [CT_EXT], is_ext=True)
DotBracketExt = Field(str, DOT_EXTS, is_ext=True)
DmsReactsExt = Field(str, [DMS_EXT], is_ext=True)
GraphExt = Field(str, GRAPH_EXTS, is_ext=True)
WebAppFileExt = Field(str, [JSON_EXT], is_ext=True)
SvgExt = Field(str, [SVG_EXT], is_ext=True)
KtsExt = Field(str, [KTS_EXT], is_ext=True)


def check_file_extension(file: pathlib.Path,
                         extensions: Iterable[str] | Field):
    if isinstance(extensions, Field):
        if not extensions.is_ext:
            raise PathValueError(f"{extensions} is not an extension field")
        extensions = extensions.options
    elif not isinstance(extensions, (tuple, list, set, dict)):
        extensions = set(extensions)
    if file.suffix not in extensions:
        raise WrongFileExtensionError(
            f"Extension of {file} is not one of {extensions}"
        )


# Path Segments ########################################################

# Segment class

class Segment(object):

    def __init__(self, segment_name: str,
                 field_types: dict[str, Field], *,
                 order: int = 0,
                 frmt: str | None = None):
        self.name = segment_name
        self.field_types = field_types
        # Verify that a field has the key EXT if and only if it is an
        # extension and is the last field in the segment.
        for i, (name, field) in enumerate(self.field_types.items(), start=1):
            if name == EXT:
                if not field.is_ext:
                    raise PathValueError(f"Field '{EXT}' is not an extension")
                if i != len(self.field_types):
                    raise PathValueError(
                        f"Extension of {self} is not the last field"
                    )
                if order != 0:
                    raise ValueError("Segments with extensions must have order "
                                     f"= 0, but {self.name} has order {order}")
            elif field.is_ext:
                raise PathValueError(f"{self} extension has name '{name}'")
        if order <= 0 and not any(ft in self.field_types for ft in [EXT, TOP]):
            raise ValueError("Segments without extensions must have order > 0, "
                             f"but {self.name} has order {order}")
        self.order = order
        # Determine the format string.
        if frmt is None:
            # Default format is to concatenate all the fields.
            frmt = "".join("{" + name + "}" for name, field
                           in self.field_types.items())
        self.frmt = frmt
        # Generate the pattern string using the format string, excluding
        # the extension (if any) because before parsing, it is removed
        # from the end of the string.
        patterns = {name: "" if field.is_ext else RE_PATTERNS[field.dtype]
                    for name, field in self.field_types.items()}
        self.ptrn = re.compile(self.frmt.format(**patterns))

    @property
    def ext_type(self):
        """ Type of the segment's file extension, or None if it has no
        file extension. """
        return self.field_types.get(EXT)

    @cached_property
    def exts(self) -> list[str]:
        """ Valid file extensions of the segment. """
        if self.ext_type is None:
            return list()
        if not self.ext_type.options:
            raise ValueError(f"{self} extension {self.ext_type} has no options")
        return self.ext_type.options

    def match_longest_ext(self, text: str):
        """ Find the longest extension of the given text that matches a
        valid file extension. If none match, return None. """
        # Iterate over the file extensions from longest to shortest.
        for ext in sorted(self.exts, key=len, reverse=True):
            if text.endswith(ext):
                # The text ends with this extension, so it must be the
                # longest valid extension in which the text ends.
                return ext
        return

    def build(self, **vals: Any):
        # Verify that a value is given for every field, with no extras.
        if (v := sorted(vals.keys())) != (f := sorted(self.field_types.keys())):
            raise PathValueError(f"{self} expected fields {f}, but got {v}")
        # Validate the value passed to every field.
        fields = {name: field.build(vals[name])
                  for name, field in self.field_types.items()}
        # Return the formatted segment.
        return self.frmt.format(**fields)

    def parse(self, text: str):
        ext = None
        if self.ext_type is not None:
            # If the segment has a file extension, then determine the
            # longest valid file extension that matches the text.
            if (ext := self.match_longest_ext(text)) is None:
                raise PathValueError(f"Segment {repr(text)} is missing a file "
                                     f"extension; expected one of {self.exts}")
            # Remove the file extension from the end of the text.
            text = text[: -len(ext)]
        # Try to parse the text (with the extension, if any, removed).
        if not (match := self.ptrn.match(text)):
            raise PathValueError(f"Could not parse fields in text {repr(text)} "
                                 f"using pattern {repr(self.ptrn)}")
        vals = list(match.groups())
        # If there is an extension field, add its value back to the end
        # of the parsed values.
        if ext is not None:
            vals.append(ext)
        # Return a dict of the names of the fields in the segment and
        # their parsed values.
        return {name: field.parse(group) for (name, field), group
                in zip(self.field_types.items(), vals, strict=True)}

    @cached_property
    def as_str(self):
        # Define the string as a cached property to speed up str(self).
        return f"{type(self).__name__} {repr(self.name)}"

    def __str__(self):
        return self.as_str


# Field names

TOP = "top"
STAGE = "stage"
CMD = "cmd"
SAMP = "sample"
REF = "ref"
REG = "reg"
BATCH = "batch"
TABLE = "table"
NCLUST = "k"
RUN = "run"
PROFILE = "profile"
GRAPH = "graph"
EXT = "ext"
STRUCT = "struct"

# Directory segments

TopSeg = Segment("top-dir", {TOP: TopField}, order=-1)
StageSeg = Segment("stage-dir", {STAGE: StageField}, order=70)
SampSeg = Segment("sample-dir", {SAMP: NameField}, order=60)
CmdSeg = Segment("command-dir", {CMD: CmdField}, order=50)
RefSeg = Segment("ref-dir", {REF: NameField}, order=30)
RegSeg = Segment("reg-dir", {REG: NameField}, order=20)

# File segments

# FASTA
FastaSeg = Segment("fasta", {REF: NameField, EXT: FastaExt})
FastaIndexSeg = Segment("fasta-index", {REF: NameField, EXT: FastaIndexExt})

# FASTQ
FastqSeg = Segment("fastq", {SAMP: NameField, EXT: FastqExt})
Fastq1Seg = Segment("fastq1", {SAMP: NameField, EXT: Fastq1Ext})
Fastq2Seg = Segment("fastq2", {SAMP: NameField, EXT: Fastq2Ext})

# Demultiplexed FASTQ
DmFastqSeg = Segment("dm-fastq", {REF: NameField, EXT: FastqExt})
DmFastq1Seg = Segment("dm-fastq1", {REF: NameField, EXT: Fastq1Ext})
DmFastq2Seg = Segment("dm-fastq2", {REF: NameField, EXT: Fastq2Ext})

# Align
XamSeg = Segment("xam", {REF: NameField, EXT: XamExt})
AlignSampleRepSeg = Segment("align-samp-rep",
                            {EXT: ReportExt},
                            frmt="align-report{ext}")
AlignRefRepSeg = Segment("align-ref-rep",
                         {REF: NameField, EXT: ReportExt},
                         frmt="{ref}__align-report{ext}")

# Relate
RefseqFileSeg = Segment("refseq-file",
                        {EXT: RefseqFileExt},
                        frmt="refseq{ext}")
ReadNamesBatSeg = Segment("names-bat",
                          {BATCH: IntField, EXT: BatchExt},
                          frmt=NAMES_STEP + "-batch-{batch}{ext}")
RelateBatSeg = Segment(f"relate-bat",
                       {BATCH: IntField, EXT: BatchExt},
                       frmt=RELATE_STEP + "-batch-{batch}{ext}")
RelateRepSeg = Segment(f"relate-rep",
                       {EXT: ReportExt},
                       frmt=RELATE_STEP + "-report{ext}")

# Mask
MaskBatSeg = Segment(f"{MASK_STEP}-bat",
                     {BATCH: IntField, EXT: BatchExt},
                     frmt=MASK_STEP + "-batch-{batch}{ext}")
MaskRepSeg = Segment("mask-rep",
                     {EXT: ReportExt},
                     frmt=MASK_STEP + "-report{ext}")

# Cluster
ClustParamsDirSeg = Segment(f"cluster-run-res-dir",
                            {},
                            frmt=CLUST_PARAMS_DIR,
                            order=10)
ClustParamsFileSeg = Segment(f"cluster-run-res",
                             {TABLE: ClustRunResultsField,
                              NCLUST: IntField,
                              RUN: IntField,
                              EXT: ClustTabExt},
                             frmt="k{k}-r{run}_{table}{ext}")
ClustBatSeg = Segment("cluster-bat",
                      {BATCH: IntField, EXT: BatchExt},
                      frmt=CLUSTER_STEP + "-batch-{batch}{ext}")
ClustRepSeg = Segment("cluster-rep",
                      {EXT: ReportExt},
                      frmt=CLUSTER_STEP + "-report{ext}")

# Deconvolve
DeconvBatSeg = Segment("deconv-bat",
                      {BATCH: IntField, EXT: BatchExt},
                      frmt="deconvolve-batch-{batch}{ext}")
DeconvRepSeg = Segment("deconv-rep", {EXT: ReportExt}, frmt="deconvolve-report{ext}")

# Table
PositionTableSeg = Segment("position-table",
                           {TABLE: PosTableField, EXT: PosTableExt},
                           frmt="{table}-position-table{ext}")
ReadTableSeg = Segment("read-table",
                       {TABLE: ReadTableField, EXT: ReadTableExt},
                       frmt="{table}-read-table{ext}")
AbundanceTableSeg = Segment("abundance-table",
                            {TABLE: AbundanceField, EXT: AbundanceExt},
                            frmt="{table}-abundance-table{ext}")

# Fold
FoldRepSeg = Segment("fold-rep",
                     {PROFILE: NameField, EXT: ReportExt},
                     frmt="{profile}__fold-report{ext}")
ConnectTableSeg = Segment("rna-ct",
                          {PROFILE: NameField, EXT: ConnectTableExt})
DotBracketSeg = Segment("rna-dot",
                        {PROFILE: NameField, EXT: DotBracketExt})
DmsReactsSeg = Segment("dms-reacts",
                       {PROFILE: NameField, EXT: DmsReactsExt})
VarnaColorSeg = Segment("varna-color",
                        {PROFILE: NameField, EXT: TextExt},
                        frmt="{profile}__varna-color{ext}")

# Draw
SvgSeg = Segment("svg", {PROFILE: NameField, STRUCT: IntField, EXT: SvgExt},
                 frmt="{profile}-{struct}{ext}")
KtsSeg = Segment("kts", {PROFILE: NameField, STRUCT: IntField, EXT: KtsExt},
                 frmt="{profile}-{struct}{ext}")

# Graphs
GraphSeg = Segment("graph", {GRAPH: NameField, EXT: GraphExt})

# Web App Export
WebAppFileSeg = Segment("webapp",
                        {SAMP: NameField, EXT: WebAppFileExt},
                        frmt="{sample}__webapp{ext}")

# Path segment patterns
CMD_DIR_SEGS = SampSeg, CmdSeg
REF_DIR_SEGS = CMD_DIR_SEGS + (RefSeg,)
REG_DIR_SEGS = REF_DIR_SEGS + (RegSeg,)
STAGE_DIR_SEGS = SampSeg, CmdSeg, StageSeg
FASTA_STAGE_SEGS = StageSeg, FastaSeg
FASTA_INDEX_DIR_STAGE_SEGS = StageSeg, RefSeg
FASTQ_SEGS = FastqSeg,
FASTQ1_SEGS = Fastq1Seg,
FASTQ2_SEGS = Fastq2Seg,
DMFASTQ_SEGS = SampSeg, DmFastqSeg
DMFASTQ1_SEGS = SampSeg, DmFastq1Seg
DMFASTQ2_SEGS = SampSeg, DmFastq2Seg
XAM_SEGS = CMD_DIR_SEGS + (XamSeg,)
XAM_STAGE_SEGS = STAGE_DIR_SEGS + (XamSeg,)
CLUST_TAB_SEGS = REG_DIR_SEGS + (ClustParamsDirSeg, ClustParamsFileSeg)
CT_FILE_SEGS = REG_DIR_SEGS + (ConnectTableSeg,)
DB_FILE_SEGS = REG_DIR_SEGS + (DotBracketSeg,)


# Paths ################################################################


# Path class

class Path(object):

    def __init__(self, *seg_types: Segment):
        # Sort the non-redundant segment types in the path from largest
        # to smallest value of their order attribute.
        self.seg_types = sorted(set(seg_types),
                                key=lambda segt: segt.order,
                                reverse=True)
        # Check for TopSeg.
        if TopSeg in self.seg_types:
            raise PathValueError(f"{TopSeg} may not be given in seg_types")
        self.seg_types.insert(0, TopSeg)
        # Check for duplicate orders.
        if max(Counter(segt.order for segt in self.seg_types).values()) > 1:
            raise ValueError(f"Got duplicate order values in {self.seg_types}")

    def build(self, **fields: Any):
        """ Return a `pathlib.Path` instance by assembling the given
        `fields` into a full path. """
        # Build the new path one segment at a time.
        segments = list()
        for seg_type in self.seg_types:
            # For each type of segment in the path, try to get the names
            # and values of all fields of the segment.
            try:
                seg_fields = {name: fields.pop(name)
                              for name in seg_type.field_types}
            except KeyError as error:
                raise PathValueError(f"Missing field for {seg_type}: {error}")
            # Generate a string representation of the segment using the
            # values of its fields, and add it to the growing path.
            segments.append(seg_type.build(**seg_fields))
        # Check whether any fields were given but not used by the path.
        if fields:
            exp = [ft for seg in self.seg_types for ft in seg.field_types]
            segs = [str(seg) for seg in self.seg_types]
            raise PathValueError(f"Unexpected fields: {fields}; expected "
                                 f"fields {exp} for segment types {segs}")
        # Assemble the segment strings into a path, and return it.
        path = pathlib.Path(*segments)
        return path

    def parse(self, path: str | pathlib.Path):
        """ Return the field names and values from a given path. """
        # Convert the given path into a canonical, absolute path.
        path = str(sanitize(path))
        # Get the field names and values one segment at a time.
        fields = dict()
        # Iterate from the deepest (last) to shallowest (first) segment.
        for seg_type in reversed(self.seg_types):
            if seg_type is TopSeg:
                # The top-most segment has been reached and must be used
                # to parse the entire remaining path.
                tail = path
            else:
                # The top-most segment of the path has not been reached.
                # Split off the deepest part of the path (tail), and
                # parse it using the current segment type.
                path, tail = os.path.split(path)
            # Verify that the entire path has not been consumed.
            if not tail:
                raise PathValueError(f"No path remains to parse {seg_type}")
            # Parse the deepest part of the path to obtain the fields,
            # and use them to update the field names and values.
            fields.update(seg_type.parse(tail))
        return fields

    @cached_property
    def as_str(self):
        # Define the string as a cached property to speed up str(self).
        return f"{type(self).__name__}: {list(map(str, self.seg_types))}"

    def __str__(self):
        return self.as_str


# mkdir/symlink/rmdir.


def mkdir_if_needed(path: pathlib.Path | str):
    """ Create a directory and log that event if it does not exist. """
    path = sanitize(path, strict=False)
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        if not path.is_dir():
            # Raise an error if the existing path is not a directory,
            # e.g. if it is a file.
            raise NotADirectoryError(path) from None
        return path
    logger.action(f"Created directory {path}")
    return path


def symlink_if_needed(link_path: pathlib.Path | str,
                      target_path: pathlib.Path | str):
    """ Make link_path a link pointing to target_path and log that event
    if it does not exist. """
    link_path = pathlib.Path(link_path)
    target_path = sanitize(target_path, strict=True)
    try:
        link_path.symlink_to(target_path)
    except FileExistsError:
        # link_path already exists, so make sure it is a symbolic link
        # that points to target_path.
        try:
            readlink = link_path.readlink()
        except OSError:
            raise OSError(f"{link_path} is not a symbolic link") from None
        if readlink != target_path:
            raise OSError(f"{link_path} is a symbolic link to {readlink}, "
                          f"not to {target_path}") from None
        return link_path
    logger.action(f"Made {link_path} a symbolic link to {target_path}")
    return link_path


def rmdir_if_needed(path: pathlib.Path | str,
                    rmtree: bool = False,
                    rmtree_ignore_errors: bool = False,
                    raise_on_rmtree_error: bool = True):
    """ Remove a directory and log that event if it exists. """
    path = sanitize(path, strict=False)
    try:
        path.rmdir()
    except FileNotFoundError:
        # The path does not exist, so there is no need to delete it.
        # FileNotFoundError is a subclass of OSError, so need to handle
        # this exception before OSError.
        logger.detail(f"Skipped removing directory {path}: does not exist")
        return path
    except NotADirectoryError:
        # Trying to rmdir() something that is not a directory should
        # always raise an error. NotADirectoryError is a subclass of
        # OSError, so need to handle this exception before OSError.
        raise
    except OSError:
        # The directory exists but could not be removed for some reason,
        # probably that it is not empty.
        if not rmtree:
            # For safety, remove directories recursively only if given
            # explicit permission to do so; if not, re-raise the error.
            raise
        try:
            shutil.rmtree(path, ignore_errors=rmtree_ignore_errors)
        except Exception as error:
            if raise_on_rmtree_error:
                raise
            # If not raising errors, then log a warning but return now
            # to avoid logging that the directory was removed.
            logger.warning(error)
            return path
    logger.action(f"Removed directory {path}")


# Path creation routines


@cache
def create_path_type(*segment_types: Segment):
    """ Create and cache a Path instance from the segment types. """
    return Path(*segment_types)


def build(*segment_types: Segment, **field_values: Any):
    """ Return a `pathlib.Path` from the given segment types and
    field values. """
    return create_path_type(*segment_types).build(**field_values)


def builddir(*segment_types: Segment, **field_values: Any):
    """ Build the path and create it on the file system as a directory
    if it does not already exist. """
    return mkdir_if_needed(build(*segment_types, **field_values))


def buildpar(*segment_types: Segment, **field_values: Any):
    """ Build a path and create its parent directory if it does not
    already exist. """
    path = build(*segment_types, **field_values)
    mkdir_if_needed(path.parent)
    return path


def randdir(parent: str | pathlib.Path | None = None,
            prefix: str = "",
            suffix: str = ""):
    """ Build a path of a new directory that does not exist and create
    it on the file system. """
    parent = sanitize(parent) if parent is not None else pathlib.Path.cwd()
    path = pathlib.Path(mkdtemp(dir=parent, prefix=prefix, suffix=suffix))
    logger.action(f"Created directory {path}")
    return path


# Path parsing routines

def get_fields_in_seg_types(*segment_types: Segment) -> dict[str, Field]:
    """ Get all fields among the given segment types. """
    fields = {field_name: field
              for segment_type in segment_types
              for field_name, field in segment_type.field_types.items()}
    return fields


def deduplicate(paths: Iterable[str | pathlib.Path], warn: bool = True):
    """ Yield the non-redundant paths. """
    total = 0
    seen = set()
    for path in map(sanitize, paths):
        total += 1
        if path in seen:
            if warn:
                logger.warning(f"Duplicate path: {path}")
        else:
            seen.add(path)
            yield path


def deduplicated(func: Callable):
    """ Decorate a Path generator to yield non-redundant paths. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        yield from deduplicate(func(*args, **kwargs))

    return wrapper


def parse(path: str | pathlib.Path, /, *segment_types: Segment):
    """ Return the fields of a path based on the segment types. """
    return create_path_type(*segment_types).parse(path)


def parse_top_separate(path: str | pathlib.Path, /, *segment_types: Segment):
    """ Return the fields of a path, and the `top` field separately. """
    field_values = parse(path, *segment_types)
    return field_values.pop(TOP), field_values


def path_matches(path: str | pathlib.Path, segments: Sequence[Segment]):
    """ Check if a path matches a sequence of path segments.

    Parameters
    ----------
    path: str | pathlib.Path
        Path of the file/directory.
    segments: Sequence[Segment]
        Sequence of path segments to check if the file matches.

    Returns
    -------
    bool
        Whether the path matches any given sequence of path segments.
    """
    # Parsing the path will succeed if and only if it matches the
    # sequence of path segments.
    try:
        parse(path, *segments)
    except PathError:
        # The path does not match this sequence of path segments.
        return False
    else:
        # The path matches this sequence of path segments.
        return True


@deduplicated
def find_files(path: str | pathlib.Path,
               segments: Sequence[Segment],
               pre_sanitize: bool = True):
    """ Yield all files that match a sequence of path segments.
    The behavior depends on what `path` is:

    - If it is a file, then yield `path` if it matches the segments;
      otherwise, yield nothing.
    - If it is a directory, then search it recursively and yield every
      matching file in the directory and its subdirectories.

    Parameters
    ----------
    path: str | pathlib.Path
        Path of a file to check or a directory to search recursively.
    segments: Sequence[Segment]
        Sequence(s) of Path segments to check if each file matches.
    pre_sanitize: bool
        Whether to sanitize the path before searching it.

    Returns
    -------
    Generator[Path, Any, None]
        Paths of files matching the segments.
    """
    if pre_sanitize:
        path = sanitize(path, strict=True)
    if path.is_file():
        # Check if the file matches the segments.
        if path_matches(path, segments):
            # If so, then yield it.
            logger.detail(f"Found file {path}")
            yield path
    else:
        # Search the directory for files matching the segments.
        logger.routine(f"Began recursively searching directory {path}")
        yield from chain(*map(partial(find_files,
                                      segments=segments,
                                      pre_sanitize=False),
                              path.iterdir()))
        logger.routine(f"Ended recursively searching directory {path}")


@deduplicated
def find_files_chain(paths: Iterable[str | pathlib.Path],
                     segments: Sequence[Segment]):
    """ Yield from `find_files` called on every path in `paths`. """
    for path in deduplicate(paths):
        try:
            yield from find_files(path, segments)
        except Exception as error:
            logger.error(error)


# Path transformation routines


def cast_path(input_path: pathlib.Path,
              input_segments: Sequence[Segment],
              output_segments: Sequence[Segment],
              **override: Any):
    """ Cast `input_path` made of `input_segments` to a new path made of
    `output_segments`.

    Parameters
    ----------
    input_path: pathlib.Path
        Input path from which to take the path fields.
    input_segments: Sequence[Segment]
        Path segments to use to determine the fields in `input_path`.
    output_segments: Sequence[Segment]
        Path segments to use to determine the fields in `output_path`.
    **override: Any
        Override and supplement the fields in `input_path`.

    Returns
    -------
    pathlib.Path
        Path comprising `output_segments` made of fields in `input_path`
        (as determined by `input_segments`).
    """
    # Extract the fields from the input path using the input segments.
    top, fields = parse_top_separate(input_path, *input_segments)
    if override:
        # Override and supplement the fields in the input path.
        fields |= override
    # Normalize the fields to comply with the output segments.
    fields = {field_name: fields[field_name]
              for field_name in get_fields_in_seg_types(*output_segments)}
    # Generate a new output path from the normalized fields.
    output_path = build(*output_segments, top=top, **fields)
    return output_path


def transpath(to_dir: str | pathlib.Path,
              from_dir: str | pathlib.Path,
              path: str | pathlib.Path,
              strict: bool = False):
    """ Return the path that would be produced by moving `path` from
    `from_dir` to `to_dir` (but do not actually move the path on the
    file system). This function does not require that any of the given
    paths exist, unless `strict` is True.

    Parameters
    ----------
    to_dir: str | pathlib.Path
        Directory to which to move `path`.
    from_dir: str | pathlib.Path
        Directory from which to move `path`; must contain `path` but not
        necessarily be the direct parent directory of `path`.
    path: str | pathlib.Path
        Path to move; can be a file or directory.
    strict: bool = False
        Require that all paths exist and contain no symbolic link loops.

    Returns
    -------
    pathlib.Path
        Hypothetical path after moving `path` from `indir` to `outdir`.
    """
    # Ensure from_dir is sanitized.
    from_dir = sanitize(from_dir, strict)
    # Find the part of the given path relative to from_dir.
    relpath = sanitize(path, strict).relative_to(from_dir)
    if relpath == pathlib.Path():
        # If the relative path is empty, then use the parent directory
        # of from_dir instead.
        return transpath(to_dir, from_dir.parent, path, strict)
    # Append the relative part of the path to to_dir.
    return sanitize(to_dir, strict).joinpath(relpath)


def transpaths(to_dir: str | pathlib.Path,
               *paths: str | pathlib.Path,
               strict: bool = False):
    """ Return all paths that would be produced by moving all paths in
    `paths` from their longest common sub-path to `to_dir` (but do not
    actually move the paths on the file system). This function does not
    require that any of the given paths exist, unless `strict` is True.

    Parameters
    ----------
    to_dir: str | pathlib.Path
        Directory to which to move every path in `path`.
    *paths: str | pathlib.Path
        Paths to move; can be files or directories. A common sub-path
        must exist among all of these paths.
    strict: bool = False
        Require that all paths exist and contain no symbolic link loops.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Hypothetical paths after moving all paths in `path` to `outdir`.
    """
    if not paths:
        # There are no paths to transplant.
        return tuple()
    # Determine the longest common sub-path of all given paths.
    common_path = os.path.commonpath([sanitize(p, strict) for p in paths])
    # Move each path from that common path to the given directory.
    return tuple(transpath(to_dir, common_path, p, strict) for p in paths)
