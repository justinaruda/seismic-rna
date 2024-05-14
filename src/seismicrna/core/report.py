from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from functools import cache
from inspect import getmembers
from logging import getLogger
from pathlib import Path
from typing import Any, Hashable, Callable

import numpy as np
from click import Option

from .arg import (opt_phred_enc,
                  opt_fastqc,
                  opt_cutadapt,
                  opt_cut_q1,
                  opt_cut_q2,
                  opt_cut_g1,
                  opt_cut_g2,
                  opt_cut_a1,
                  opt_cut_a2,
                  opt_cut_discard_trimmed,
                  opt_cut_discard_untrimmed,
                  opt_cut_o,
                  opt_cut_e,
                  opt_cut_indels,
                  opt_cut_nextseq,
                  opt_cut_m,
                  opt_bt2_d,
                  opt_bt2_r,
                  opt_bt2_dpad,
                  opt_bt2_orient,
                  opt_bt2_i,
                  opt_bt2_x,
                  opt_bt2_s,
                  opt_bt2_l,
                  opt_bt2_gbar,
                  opt_bt2_un,
                  opt_bt2_discordant,
                  opt_bt2_mixed,
                  opt_bt2_dovetail,
                  opt_bt2_contain,
                  opt_bt2_local,
                  opt_cram,
                  opt_min_reads,
                  opt_min_mapq,
                  opt_min_ncov_read,
                  opt_overhangs,
                  opt_fold_temp,
                  opt_fold_md,
                  opt_fold_mfe,
                  opt_fold_max,
                  opt_fold_percent,
                  opt_quantile,
                  opt_quick_unbias,
                  opt_quick_unbias_thresh,
                  opt_min_finfo_read,
                  opt_min_mut_gap,
                  opt_min_ninfo_pos,
                  opt_max_fmut_pos,
                  opt_ambindel,
                  opt_em_runs,
                  opt_em_thresh,
                  opt_min_em_iter,
                  opt_max_em_iter,
                  opt_max_fmut_read,
                  opt_max_clusters,
                  opt_exclude_gu,
                  opt_exclude_polya,
                  opt_discontig,
                  opt_min_phred)
from .io import FileIO, ReadBatchIO, RefIO
from .rel import HalfRelPattern
from .version import __version__
from .write import need_write, write_mode

logger = getLogger(__name__)


# Field class

class Field(object):
    """ Field of a report. """

    __slots__ = "key", "title", "dtype", "default", "iconv", "oconv"

    def __init__(self,
                 key: str,
                 title: str,
                 dtype: type,
                 default: Any | None = None, *,
                 iconv: Callable[[Any], Any] | None = None,
                 oconv: Callable[[Any], Any] | None = None):
        """
        Parameters
        ----------
        key: str
            Key under which the field is stored as an attribute of the
            Report instance.
        title: str
            Title by which the field is identified in the report file.
        dtype: type
            Data type of the field
        default: Any
            Default value of the field
        iconv: Callable[[Any], Any] | None = None
            Convert an input value to the right type for instantiation.
            If omitted, just cast the value to the proper data type.
        oconv: Callable[[Any], Any] | None = None
            Convert an output value to the right type for exporting.
            If omitted, export the value with no change.
        """
        self.key = key
        self.title = title
        self.dtype = dtype
        self.default = default
        self.iconv = iconv if iconv is not None else self.dtype
        self.oconv = oconv

    def __str__(self):
        return f"{type(self).__name__} {repr(self.title)} ({self.key})"


class OptionField(Field):
    """ Field based on a command line option. """

    def __init__(self, option: Option, **kwargs):
        """
        Parameters
        ----------
        option: click.Option
            Option from which to make the field.
        **kwargs
            Additional keyword arguments passed to `Field`.
        """
        super().__init__(option.name,
                         option.help,
                         option.type,
                         option.default,
                         **kwargs)


# Field calculation functions

# Note that each function takes a single argument: a Report instance.
# So these functions could be implemented as Report instance methods.
# But this implementation could cause confusion because no one Report
# class can work with all these methods.


def calc_dt_minutes(began: datetime, ended: datetime):
    """ Calculate the time taken in minutes. """
    delta = ended - began
    minutes = (delta.seconds + 1e-6 * delta.microseconds) / 60.
    if minutes < 0.:
        raise ValueError(f"Time taken must be positive, but got {minutes} min")
    return minutes


def calc_taken(report: Report):
    """ Calculate the time taken in minutes. """
    return calc_dt_minutes(report.get_field(TimeBeganF),
                           report.get_field(TimeEndedF))


# Field definitions

DATETIME_FORMAT = "%Y-%m-%d at %H:%M:%S"
DECIMAL_PRECISION = 3  # general precision for decimals
PERC_VEC_PRECISION = 1
TIME_TAKEN_PRECISION = 2


def iconv_int_keys(mapping: dict[Any, Any]):
    return {int(key): value for key, value in mapping.items()}


def iconv_array_int(nums: list[int]):
    return np.asarray(nums, dtype=int)


def oconv_array_int(nums: np.ndarray):
    return list(map(int, nums))


def iconv_dict_str_int(mapping: dict[Any, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in mapping.items()}


def iconv_dict_str_dict_int_dict_int_int(
        mapping: dict[Any, dict[Any, dict[Any, Any]]]
) -> dict[str, dict[int, dict[int, int]]]:
    return {str(key1): {int(key2): {int(key3): int(val3)
                                    for key3, val3 in val2.items()}
                        for key2, val2 in val1.items()}
            for key1, val1 in mapping.items()}


@cache
def get_oconv_float(precision: int = DECIMAL_PRECISION):
    def oconv_float(num: float):
        return round(num, precision)

    return oconv_float


@cache
def get_oconv_list_float(precision: int = DECIMAL_PRECISION):
    oconv_float = get_oconv_float(precision)

    def oconv_list_float(nums: list[float]) -> list[float]:
        return list(map(oconv_float, nums))

    return oconv_list_float


@cache
def get_oconv_array_float(precision: int = DECIMAL_PRECISION):
    oconv_list_float = get_oconv_list_float(precision)

    def oconv_array_float(nums: np.array):
        return oconv_list_float(nums.tolist())

    return oconv_array_float


@cache
def get_oconv_dict_float(precision: int = DECIMAL_PRECISION):
    oconv_float = get_oconv_float(precision)

    def oconv_dict_float(dnum: dict[Hashable, float]) -> dict[Hashable, float]:
        return {d: oconv_float(num) for d, num in dnum.items()}

    return oconv_dict_float


@cache
def get_oconv_dict_list_float(precision: int = DECIMAL_PRECISION):
    oconv_list_float = get_oconv_list_float(precision)

    def oconv_dict_list_float(dnums: dict[Hashable, list[float]]
                              ) -> dict[Hashable, list[float]]:
        return {d: oconv_list_float(nums) for d, nums in dnums.items()}

    return oconv_dict_list_float


def iconv_datetime(text: str):
    return datetime.strptime(text, DATETIME_FORMAT)


def oconv_datetime(dtime: datetime):
    return dtime.strftime(DATETIME_FORMAT)


# General fields
VersionF = Field("version", "version of SEISMIC-RNA", str, __version__)
BranchesF = Field("branches", "branches", list, list())
SampleF = Field("sample", "sample", str)
RefF = Field("ref", "reference", str)
SectF = Field("sect", "section", str)
End5F = Field("end5", "section 5' end", int)
End3F = Field("end3", "section 3' end", int)
MinReadsF = OptionField(opt_min_reads)
TimeBeganF = Field("began", "time began", datetime,
                   iconv=iconv_datetime, oconv=oconv_datetime)
TimeEndedF = Field("ended", "time ended", datetime,
                   iconv=iconv_datetime, oconv=oconv_datetime)
TimeTakenF = Field("taken", "time taken (minutes)", float, calc_taken,
                   oconv=get_oconv_float(TIME_TAKEN_PRECISION))

# Align fields
IsDemultF = Field("demultiplexed", "Use demultiplexed mode", bool)
IsPairedEndF = Field("paired_end", "Use paired-end mode", bool)
PhredEncF = OptionField(opt_phred_enc)
UseFastqcF = OptionField(opt_fastqc)
UseCutadaptF = OptionField(opt_cutadapt)
CutadaptQ1 = OptionField(opt_cut_q1)
CutadaptQ2 = OptionField(opt_cut_q2)
CutadaptG1 = OptionField(opt_cut_g1)
CutadaptA1 = OptionField(opt_cut_a1)
CutadaptG2 = OptionField(opt_cut_g2)
CutadaptA2 = OptionField(opt_cut_a2)
CutadaptOverlap = OptionField(opt_cut_o)
CutadaptErrors = OptionField(opt_cut_e)
CutadaptIndels = OptionField(opt_cut_indels)
CutadaptNextSeq = OptionField(opt_cut_nextseq)
CutadaptNoTrimmed = OptionField(opt_cut_discard_trimmed)
CutadaptNoUntrimmed = OptionField(opt_cut_discard_untrimmed)
CutadaptMinLength = OptionField(opt_cut_m)
Bowtie2Local = OptionField(opt_bt2_local)
Bowtie2Discord = OptionField(opt_bt2_discordant)
Bowtie2Dovetail = OptionField(opt_bt2_dovetail)
Bowtie2Contain = OptionField(opt_bt2_contain)
Bowtie2Mixed = OptionField(opt_bt2_mixed)
Bowtie2Un = OptionField(opt_bt2_un)
Bowtie2ScoreMin = Field("bt2_score_min",
                        "minimum alignment score for Bowtie2",
                        str)
Bowtie2MinLengthF = OptionField(opt_bt2_i)
Bowtie2MaxLengthF = OptionField(opt_bt2_x)
Bowtie2GBarF = OptionField(opt_bt2_gbar)
Bowtie2SeedLength = OptionField(opt_bt2_l)
Bowtie2SeedInterval = OptionField(opt_bt2_s)
Bowtie2ExtTries = OptionField(opt_bt2_d)
Bowtie2Reseed = OptionField(opt_bt2_r)
Bowtie2Dpad = OptionField(opt_bt2_dpad)
Bowtie2Orient = OptionField(opt_bt2_orient)
MinMapQualF = OptionField(opt_min_mapq)
CramOutF = OptionField(opt_cram)
AlignReadsInitF = Field("align_reads_init", "number of reads in FASTQ", int)
ReadsTrimF = Field("reads_trim", "number of reads after trimming", int)
ReadsAlignF = Field("reads_align", "number of reads after alignment", dict,
                    iconv=iconv_dict_str_int)
ReadsDedupF = Field("reads_filter",
                    "number of reads after filtering",
                    dict,
                    iconv=iconv_dict_str_int)
ReadsRefsF = Field("reads_refs",
                   "number of reads aligned by reference",
                   dict,
                   iconv=iconv_dict_str_int)

# Relate fields
NumReadsRelF = Field("n_reads_rel", "number of reads", int)
NumBatchF = Field("n_batches", "number of batches", int)
ChecksumsF = Field("checksums", "MD5 checksums of batches", dict)
RefseqChecksumF = Field("refseq_checksum",
                        "MD5 checksum of reference sequence",
                        str)
AmbindelF = OptionField(opt_ambindel)
OverhangsF = OptionField(opt_overhangs)
MinPhredF = OptionField(opt_min_phred)

# Pool fields
PooledSamplesF = Field("pooled_samples", "pooled samples", list)

# Mask fields
CountMutsF = Field("count_muts",
                   "count as mutations",
                   HalfRelPattern,
                   iconv=HalfRelPattern.from_report_format,
                   oconv=HalfRelPattern.to_report_format)
CountRefsF = Field("count_refs",
                   "count as matches",
                   HalfRelPattern,
                   iconv=HalfRelPattern.from_report_format,
                   oconv=HalfRelPattern.to_report_format)
ExclPolyAF = OptionField(opt_exclude_polya)
ExclGUF = OptionField(opt_exclude_gu)
ExclUserPosF = Field("exclude_pos",
                     "exclude positions from a list",
                     np.ndarray,
                     iconv=iconv_array_int,
                     oconv=oconv_array_int)
MinNInfoPosF = OptionField(opt_min_ninfo_pos)
MaxFMutPosF = OptionField(opt_max_fmut_pos)
MinNCovReadF = OptionField(opt_min_ncov_read)
DiscontigF = OptionField(opt_discontig)
MinMutGapF = OptionField(opt_min_mut_gap)
QuickUnbiasF = OptionField(opt_quick_unbias)
QuickUnbiasThreshF = OptionField(opt_quick_unbias_thresh)
MinFInfoReadF = OptionField(opt_min_finfo_read)
MaxFMutReadF = OptionField(opt_max_fmut_read)
PosCutPolyAF = Field("pos_polya",
                     "positions cut for being poly(A) sequences",
                     np.ndarray,
                     iconv=iconv_array_int,
                     oconv=oconv_array_int)
PosCutGUF = Field("pos_gu",
                  "positions cut for being G/U bases",
                  np.ndarray,
                  iconv=iconv_array_int,
                  oconv=oconv_array_int)
PosCutListF = Field("pos_list",
                    "positions cut for being listed",
                    np.ndarray,
                    iconv=iconv_array_int,
                    oconv=oconv_array_int)
PosCutLoInfoF = Field("pos_min_ninfo",
                      "positions cut for insufficient information",
                      np.ndarray,
                      iconv=iconv_array_int,
                      oconv=oconv_array_int)
PosCutHiMutF = Field("pos_max_fmut",
                     "positions cut for excessive mutations",
                     np.ndarray,
                     iconv=iconv_array_int,
                     oconv=oconv_array_int)
PosKeptF = Field("pos_kept",
                 "positions kept",
                 np.ndarray,
                 iconv=iconv_array_int,
                 oconv=oconv_array_int)
NumPosInitF = Field("n_pos_init",
                    "number of positions in total",
                    int)
NumPosCutPolyAF = Field("n_pos_polya",
                        "number of positions cut for being poly(A) sequences",
                        int)
NumPosCutGUF = Field("n_pos_gu",
                     "number of positions cut for being G/U bases",
                     int)
NumPosCutListF = Field("n_pos_user",
                       "number of positions cut for being listed",
                       int)
NumPosCutLoInfoF = Field("n_pos_min_ninfo",
                         "number of positions cut for insufficient information",
                         int)
NumPosCutHiMutF = Field("n_pos_max_fmut",
                        "number of positions cut for excessive mutations",
                        int)
NumPosKeptF = Field("n_pos_kept",
                    "number of positions kept",
                    int)
NumReadsInitF = Field("n_reads_init",
                      "number of reads in total",
                      int)
NumReadsLoNCovF = Field("n_reads_min_ncov",
                        "number of reads cut for insufficient coverage",
                        int)
NumDiscontigF = Field("n_reads_discontig",
                      "number of reads cut for discontiguous mates",
                      int)
NumReadsLoInfoF = Field("n_reads_min_finfo",
                        "number of reads cut for insufficient information",
                        int)
NumReadsHiMutF = Field("n_reads_max_fmut",
                       "number of reads cut for excessive mutations",
                       int)
NumReadsCloseMutF = Field("n_reads_min_gap",
                          "number of reads cut for two mutations too close",
                          int)
NumReadsKeptF = Field("n_reads_kept",
                      "number of reads kept",
                      int)

# Cluster fields

NumUniqReadKeptF = Field("n_uniq_reads",
                         "number of unique reads",
                         int)
MinIterClustF = OptionField(opt_min_em_iter)
MaxIterClustF = OptionField(opt_max_em_iter)
ClustConvThreshF = OptionField(opt_em_thresh)
MaxClustsF = OptionField(opt_max_clusters)
ClustNumRunsF = OptionField(opt_em_runs)
ClustsConvF = Field("converged",
                    "iterations until convergence per run",
                    dict,
                    dict(),
                    iconv=iconv_int_keys)
ClustsLogLikesF = Field("log_likes",
                        "log likelihood per run",
                        dict,
                        dict(),
                        iconv=iconv_int_keys,
                        oconv=get_oconv_dict_list_float())
ClustsRMSDsF = Field("clusts_rmsds",
                     "NRMSD from run 0",
                     dict,
                     dict(),
                     iconv=iconv_int_keys,
                     oconv=get_oconv_dict_list_float())
ClustsMeanRsF = Field("clusts_meanr",
                      "Pearson correlation with run 0",
                      dict,
                      dict(),
                      iconv=iconv_int_keys,
                      oconv=get_oconv_dict_list_float())
ClustsBicF = Field("bic",
                   "Bayesian information criterion per order",
                   dict,
                   iconv=iconv_int_keys,
                   oconv=get_oconv_dict_float())
NumClustsF = Field("best_order", "optimal number of clusters", int)

# Join fields

JoinedSectionsF = Field("joined_sections", "joined sections", list)
JoinedClustersF = Field("joined_clusters",
                        "joined clusters",
                        dict,
                        iconv=iconv_dict_str_dict_int_dict_int_int)

# Fold fields

ProfileF = Field("profile", "profile", str)
Quantile = OptionField(opt_quantile)
FoldTempF = OptionField(opt_fold_temp)
FoldMaxDistF = OptionField(opt_fold_md)
FoldMinFreeEnergyF = OptionField(opt_fold_mfe)
FoldMaxStructsF = OptionField(opt_fold_max)
FoldPercent = OptionField(opt_fold_percent)


# Field managing functions

@cache
def fields():
    return [member for _, member in getmembers(sys.modules[__name__])
            if isinstance(member, Field)]


@cache
def field_keys() -> dict[str, Field]:
    keys = dict()
    for field in fields():
        if field.key:
            if field.key in keys:
                raise ValueError(f"Repeated field key: {repr(field.key)}")
            keys[field.key] = field
    return keys


@cache
def field_titles() -> dict[str, Field]:
    titles = dict()
    for field in fields():
        if field.title:
            if field.title in titles:
                raise ValueError(f"Repeated field title: {repr(field.title)}")
            titles[field.title] = field
    return titles


def lookup_key(key: str):
    """ Get a field by its key. """
    try:
        return field_keys()[key]
    except KeyError:
        raise ValueError(f"Invalid field key: {repr(key)}")


def lookup_title(title: str):
    """ Get a field by its title. """
    if not title:
        raise ValueError("Got blank title for field")
    try:
        return field_titles()[title]
    except KeyError:
        raise ValueError(f"Invalid field title: {repr(title)}")


def key_to_title(key: str):
    """ Map a field's key to its title. """
    return lookup_key(key).title


def default_key(key: str):
    """ Get the default value of a field by its key. """
    if (default := lookup_key(key).default) is None:
        raise ValueError(f"Field {repr(key_to_title(key))} has no default")
    return default


# Report classes

class Report(FileIO, ABC):
    """ Abstract base class for a report from a step. """

    @classmethod
    @abstractmethod
    def fields(cls):
        """ All fields of the report. """
        return [BranchesF, TimeBeganF, TimeEndedF, TimeTakenF, VersionF]

    @classmethod
    @cache
    def field_keys(cls):
        """ Keys of all fields of the report. """
        return [field.key for field in cls.fields()]

    @classmethod
    def from_dict(cls, odata: dict[str, Any]):
        """ Convert a dict of raw values (keyed by the titles of their
        fields) into a dict of encoded values (keyed by the keys of
        their fields), from which a new Report is instantiated. """
        if not isinstance(odata, dict):
            raise TypeError(f"Expected dict, but got {type(odata).__name__}")
        # Read every raw value, keyed by the title of its field.
        idata = dict()
        for title, value in odata.items():
            # Get the field corresponding to the title.
            try:
                field = lookup_title(title)
            except ValueError as error:
                logger.warning(error)
            else:
                # Cast the value to the input type; key it by the field.
                idata[field.key] = field.iconv(value)
        # Instantiate and return a new Report from the values.
        return cls(**idata)

    @classmethod
    def load(cls, file: Path) -> Report:
        with open(file) as f:
            report = cls.from_dict(json.load(f))
        # Ensure that the path-related fields in the JSON data match the
        # actual path of the JSON file.
        top, path_fields = cls.parse_path(file)
        for key, value in report.path_field_values().items():
            if value != path_fields.get(key):
                raise ValueError(f"Got different values for {repr(key)} "
                                 f"in path ({repr(path_fields.get(key))}) "
                                 f"and contents ({repr(value)}) of {file}")
        return report

    @classmethod
    def _auto_default_fields(cls):
        return [BranchesF, TimeTakenF, VersionF]

    @classmethod
    def _auto_init_kwargs(cls, **kwargs):
        """ Automatic keyword arguments for __init__. """
        return {field.key: field.default
                for field in cls._auto_default_fields()} | kwargs

    def __init__(self, **kwargs: Any | Callable[[Report], Any]):
        kwargs = self._auto_init_kwargs(**kwargs)
        defaulted = dict()
        for key in self.field_keys():
            # Try to get the value of the field from the report.
            try:
                value = kwargs.pop(key)
            except KeyError:
                # If the report file is missing that field (e.g. because
                # it came from a different version of SEISMIC-RNA), then
                # for cross-version compatibility, use the default value
                # of the field.
                value = default_key(key)
                defaulted[key_to_title(key)] = value
            if callable(value):
                # If the value of the keyword argument is callable, then
                # it must accept one argument -- self -- and return the
                # value of the attribute.
                value = value(self)
            setattr(self, key, value)
        if kwargs:
            # If the report file has extra fields (e.g. because it came
            # from a different version of SEISMIC-RNA), then just log a
            # warning and ignore the extra fields (to make different
            # versions compatible).
            logger.warning(f"Got extra fields for {type(self).__name__}: "
                           f"{list(kwargs)}")
        if defaulted:
            # If the report file was missing keyword arguments that have
            # default values, AND if parsing the report file succeeded,
            # then warn about the default values.
            logger.warning(f"Missing fields for {type(self).__name__} "
                           f"and using defaults: {defaulted}")

    def get_field(self, field: Field, missing_ok: bool = False):
        """ Return the value of a field of the report using the field
        instance directly, not its key. """
        try:
            return getattr(self, field.key)
        except AttributeError:
            if missing_ok:
                return None
            raise

    def to_dict(self):
        """ Return a dict of raw values of the fields, keyed by the
        titles of their fields. """
        odata = dict()
        for key in self.field_keys():
            field = lookup_key(key)
            # Output only the fields with non-blank titles.
            value = self.get_field(field)
            if field.oconv is not None:
                # Convert the value to the proper output value.
                value = field.oconv(value)
            odata[field.title] = value
        return odata

    def save(self, top: Path, force: bool = False):
        """ Save the report to a JSON file. """
        text = json.dumps(self.to_dict(), indent=4)
        save_path = self.get_path(top)
        if need_write(save_path, force):
            with open(save_path, write_mode(force)) as f:
                f.write(text)
            logger.info(f"Wrote {self} to {save_path}")
        return save_path

    def __setattr__(self, key: str, value: Any):
        """ Validate the attribute name and value before setting it. """
        if key not in self.field_keys():
            raise ValueError(
                f"Invalid field for {type(self).__name__}: {repr(key)}"
            )
        super().__setattr__(key, value)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.to_dict() == other.to_dict()


class RefseqReport(Report, RefIO, ABC):
    """ Report associated with a reference sequence file. """

    @classmethod
    @abstractmethod
    def fields(cls):
        return [RefseqChecksumF] + super().fields()


class BatchedReport(Report, ABC):
    """ Report with a number of data batches (one file per batch). """

    @classmethod
    @abstractmethod
    def fields(cls):
        return [NumBatchF, ChecksumsF] + super().fields()

    @classmethod
    @abstractmethod
    def _batch_types(cls) -> tuple[type[ReadBatchIO], ...]:
        """ Type(s) of batch(es) for the report. """

    @classmethod
    @cache
    def batch_types(cls) -> dict[str, type[ReadBatchIO]]:
        """ Type(s) of batch(es) for the report, keyed by name. """
        return {batch_type.btype(): batch_type
                for batch_type in cls._batch_types()}

    @classmethod
    def get_batch_type(cls, btype: str | None = None) -> type[ReadBatchIO]:
        """ Return a valid type of batch based on its name. """
        if btype is None:
            batch_types = list(cls.batch_types().values())
            if (ntypes := len(batch_types)) != 1:
                raise ValueError(f"btype is optional only if there is exactly "
                                 f"one type of batch, but got {ntypes} types")
            return batch_types[0]
        return cls.batch_types()[btype]


class BatchedRefseqReport(BatchedReport, RefseqReport, ABC):
    """ Convenience class used as a base for several Report classes. """

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
