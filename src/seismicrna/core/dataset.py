from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterable

from . import path
from .batch import MutsBatch, RegionMutsBatch, ReadBatch, list_batch_nums
from .io import MutsBatchIO, ReadBatchIO
from .logs import logger
from .rel import RelPattern
from .report import (DATETIME_FORMAT,
                     SampleF,
                     RefF,
                     RegF,
                     TimeEndedF,
                     NumBatchF,
                     ChecksumsF,
                     PooledSamplesF,
                     JoinedRegionsF,
                     Report,
                     BatchedReport)
from .seq import DNA, Region, unite


class ReversedTimeStampError(RuntimeError):
    """ A dataset has a timestamp that is earlier than a dataset that
    should have been written before it. """


class MissingBatchError(RuntimeError):
    """ A dataset does not have a batch of a given type and number. """


class MissingBatchTypeError(MissingBatchError):
    """ A dataset does not have a batch of a given type. """


class FailedToLoadDatasetError(RuntimeError):
    """ A batch failed to load. """


class Dataset(ABC):
    """ Dataset comprising batches of data. """

    @classmethod
    @abstractmethod
    def get_report_type(cls) -> type[Report]:
        """ Type of report. """

    def __init__(self, report_file: Path, verify_times: bool = True):
        self.report_file = report_file
        self.verify_times = verify_times
        # Load the report here so that if the report does not have the
        # expected fields, an error will be raised inside __init__.
        # Doing so is essential for LoadFunction to determine which type
        # of dataset to load, since it bases that decision on whether
        # __init__ succeeds or raises an error.
        self.report = self.get_report_type().load(self.report_file)

    @cached_property
    def top(self) -> Path:
        """ Top-level directory of the dataset. """
        top, _ = self.get_report_type().parse_path(self.report_file)
        return top

    @property
    def sample(self) -> str:
        """ Name of the sample. """
        return self.report.get_field(SampleF)

    @property
    def ref(self) -> str:
        """ Name of the reference. """
        return self.report.get_field(RefF)

    @property
    @abstractmethod
    def pattern(self) -> RelPattern | None:
        """ Pattern of mutations to count. """

    @cached_property
    @abstractmethod
    def ks(self) -> list[int]:
        """ Numbers of clusters. """

    @cached_property
    @abstractmethod
    def best_k(self) -> int:
        """ Best number of clusters. """

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """ Time at which the data were written. """

    @property
    def dir(self) -> Path:
        """ Directory containing the dataset. """
        return self.report_file.parent

    @property
    @abstractmethod
    def data_dirs(self) -> list[Path]:
        """ All directories containing data for the dataset. """

    @property
    @abstractmethod
    def num_batches(self) -> int:
        """ Number of batches. """

    @property
    def batch_nums(self):
        """ Numbers of the batches. """
        return list_batch_nums(self.num_batches)

    @abstractmethod
    def get_batch(self, batch_num: int) -> ReadBatch:
        """ Get a specific batch of data. """

    def iter_batches(self):
        """ Yield each batch. """
        for batch_num in self.batch_nums:
            yield self.get_batch(batch_num)

    @cached_property
    def num_reads(self):
        """ Number of reads in the dataset. """
        return sum(batch.num_reads for batch in self.iter_batches())

    def link_data_dirs_to_tmp(self, tmp_dir: Path):
        """ Make links to a dataset in a temporary directory. """
        for data_dir in self.data_dirs:
            tmp_data_dir = path.transpath(tmp_dir,
                                          self.top,
                                          data_dir,
                                          strict=True)
            path.mkdir_if_needed(tmp_data_dir.parent)
            path.symlink_if_needed(tmp_data_dir, data_dir)

    def __str__(self):
        return f"{type(self).__name__} in {self.report_file}"


class UnbiasDataset(Dataset, ABC):
    """ Dataset with attributes for correcting observer bias. """

    def __init__(self,
                 *args,
                 masked_read_nums: dict[[int, list]] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_read_nums = masked_read_nums

    @property
    @abstractmethod
    def min_mut_gap(self) -> int:
        """ Minimum gap between two mutations. """

    @property
    @abstractmethod
    def quick_unbias(self) -> bool:
        """ Use the quick heuristic for unbiasing. """

    @property
    @abstractmethod
    def quick_unbias_thresh(self) -> float:
        """ Consider mutation rates less than or equal to this threshold
        to be 0 when using the quick heuristic for unbiasing. """


class RegionDataset(Dataset, ABC):
    """ Dataset with a known reference sequence and region. """

    @property
    @abstractmethod
    def refseq(self) -> DNA:
        """ Sequence of the reference. """

    @property
    def reflen(self):
        """ Length of the reference sequence. """
        return len(self.refseq)

    @cached_property
    @abstractmethod
    def region(self) -> Region:
        """ Region of the dataset. """


class MutsDataset(RegionDataset, ABC):
    """ Dataset with a known region and explicit mutational data. """

    @abstractmethod
    def get_batch(self, batch_num: int) -> RegionMutsBatch:
        # Redeclare this method (already declared by Dataset) for the
        # sole purpose of enabling type linters to determine that
        # MutsDataset.get_batch() returns a RegionMutsBatch object
        # instead of plain ReadBatch objects.
        pass

    def get_batch_count_all(self, batch_num: int, **kwargs):
        """ Calculate the counts for a specific batch of data. """
        return self.get_batch(batch_num).count_all(**kwargs)

    def iter_batches(self):
        # Reimplement this method (already implemented by Dataset) for
        # the sole purpose of enabling type linters to determine that
        # MutsDataset.iter_batches() yields RegionMutsBatch objects
        # instead of plain ReadBatch objects.
        for batch_num in self.batch_nums:
            yield self.get_batch(batch_num)


class LoadFunction(object):
    """ Function to load a dataset. """

    def __init__(self, data_type: type[Dataset], /, *more_types: type[Dataset]):
        self.dataset_types = (data_type,) + more_types

    def _dataset_type_consensus(self,
                                method: Callable[[type[Dataset]], Any],
                                what: str):
        """ Get the consensus value among all types of dataset. """
        value0 = method(self.dataset_types[0])
        for dataset_type in self.dataset_types[1:]:
            assert method(dataset_type) == value0
        return value0

    @cached_property
    def report_path_seg_types(self):
        """ Segment types of the report file path. """
        return self._dataset_type_consensus(
            lambda dt: dt.get_report_type().seg_types(),
            "sequence of path segment types"
        )

    @cached_property
    def report_path_auto_fields(self):
        """ Automatic field values of the report file path. """
        return self._dataset_type_consensus(
            lambda dt: dt.get_report_type().auto_fields(),
            "automatic path fields of the report type"
        )

    def is_dataset_type(self, dataset: Dataset):
        """ Whether the dataset is one of the loadable types. """
        return any(isinstance(dataset, dataset_type)
                   for dataset_type in self.dataset_types)

    def __call__(self, report_file: Path, **kwargs):
        """ Load a dataset from the report file. """
        # Try to load the report file using each type of dataset.
        errors = dict()
        for dataset_type in self.dataset_types:
            try:
                # Return the first dataset type that works.
                return dataset_type(report_file, **kwargs)
            except FileNotFoundError:
                # Re-raise FileNotFoundError because if the report file
                # does not exist, then no dataset type can load it.
                raise
            except ReversedTimeStampError:
                # Re-raise ReversedTimeStampError because if the report
                # file's timestamp is earlier than that of one of its
                # constituents, then no dataset type can load it.
                raise
            except Exception as error:
                # If a type fails for any other reason, then record the
                # error silently.
                errors[dataset_type.__name__] = error
        # If all dataset types failed, then raise an error.
        errmsg = "\n".join(f"{type_name}: {error}"
                           for type_name, error in errors.items())
        raise FailedToLoadDatasetError(
            f"{self} failed to load {report_file}:\n{errmsg}"
        )

    def __str__(self):
        names = ", ".join(dataset_type.__name__
                          for dataset_type in self.dataset_types)
        return f"{type(self).__name__}({names})"


class LoadedDataset(Dataset, ABC):
    """ Dataset created by loading directly from a Report. """

    @classmethod
    @abstractmethod
    def get_batch_type(cls) -> type[ReadBatchIO | MutsBatchIO]:
        """ Type of batch. """

    @classmethod
    @abstractmethod
    def get_report_type(cls) -> type[BatchedReport]:
        """ Type of report. """

    @classmethod
    def get_btype_name(cls):
        """ Name of the type of batch. """
        return cls.get_batch_type().btype()

    @property
    def timestamp(self):
        return self.report.get_field(TimeEndedF)

    @property
    def num_batches(self):
        return self.report.get_field(NumBatchF)

    @property
    def data_dirs(self):
        return [self.dir]

    def get_batch_path(self, batch: int):
        """ Get the path to a batch of a specific number. """
        fields = self.report.path_field_values(self.top,
                                               self.get_batch_type().auto_fields())
        return self.get_batch_type().build_path(batch=batch, **fields)

    def get_batch_checksum(self, batch: int):
        """ Get the checksum of a specific batch from the report. """
        checksums = self.report.get_field(ChecksumsF)
        try:
            return checksums[self.get_btype_name()][batch]
        except KeyError:
            # Report does not have checksums for this type of batch.
            raise MissingBatchTypeError(self.get_batch_type())
        except IndexError:
            # Report does not have a checksum for this batch number.
            raise MissingBatchError(batch)

    def get_batch(self, batch_num: int) -> ReadBatchIO | MutsBatchIO:
        batch = self.get_batch_type().load(
            self.get_batch_path(batch_num),
            checksum=self.get_batch_checksum(batch_num)
        )
        assert batch.batch == batch_num
        return batch


class MergedDataset(Dataset, ABC):
    """ Dataset made by merging one or more constituent datasets. """

    @classmethod
    @abstractmethod
    def get_dataset_load_func(cls) -> LoadFunction:
        """ Function to load one constituent dataset. """

    @cached_property
    @abstractmethod
    def datasets(self) -> list[Dataset]:
        """ Constituent datasets that were merged. """

    @cached_property
    def data_dirs(self):
        return [self.dir] + [dataset_dir
                             for dataset in self.datasets
                             for dataset_dir in dataset.data_dirs]

    def _list_dataset_attr(self, name: str, *subnames: str):
        """ Get a list of an attribute for each dataset. """
        values = list()
        for dataset in self.datasets:
            value = getattr(dataset, name)
            for subname in subnames:
                value = getattr(value, subname)
            values.append(value)
        return values

    def _get_common_attr(self, name: str, *subnames):
        """ Get a common attribute among datasets. """
        values = list()
        for value in self._list_dataset_attr(name, *subnames):
            if value not in values:
                values.append(value)
        if len(values) != 1:
            raise ValueError(f"{type(self).__name__} expected exactly 1 value "
                             f"for attribute {name}, but got {values}")
        return values[0]

    @cached_property
    def pattern(self):
        return self._get_common_attr("pattern")

    @cached_property
    def timestamp(self):
        # Use the time of the most recent constituent dataset.
        return max(dataset.timestamp for dataset in self.datasets)


class MergedRegionDataset(MergedDataset, RegionDataset, ABC):

    @cached_property
    def refseq(self):
        return self._get_common_attr("refseq")


class MergedUnbiasDataset(MergedDataset, UnbiasDataset, ABC):
    """ MergedDataset with attributes for correcting observer bias. """

    @property
    def min_mut_gap(self):
        return self._get_common_attr("min_mut_gap")

    @property
    def quick_unbias(self):
        return self._get_common_attr("quick_unbias")

    @property
    def quick_unbias_thresh(self):
        return self._get_common_attr("quick_unbias_thresh")


class TallDataset(MergedDataset, ABC):
    """ Dataset made by vertically pooling other datasets from one or
    more samples aligned to the same reference sequence. """

    @cached_property
    def datasets(self):
        # Determine the sample name and pooled samples from the report.
        pooled_samples = self.report.get_field(PooledSamplesF)
        # Determine the report file for each sample in the pool.
        load_func = self.get_dataset_load_func()
        sample_report_files = [path.cast_path(
            self.report_file,
            self.get_report_type().seg_types(),
            load_func.report_path_seg_types,
            **load_func.report_path_auto_fields,
            sample=sample
        ) for sample in pooled_samples]
        if not sample_report_files:
            raise ValueError(f"{self} got no datasets")
        return list(map(self.get_dataset_load_func(),
                        sample_report_files))

    @cached_property
    def samples(self) -> list[str]:
        """ Names of all samples in the pool. """
        return self._list_dataset_attr("sample")

    @cached_property
    def nums_batches(self) -> list[int]:
        """ Number of batches in each dataset in the pool. """
        return self._list_dataset_attr("num_batches")

    @cached_property
    def num_batches(self):
        return sum(self.nums_batches)

    def _translate_batch_num(self, batch_num: int):
        """ Translate a batch number into the numbers of the dataset and
        of the batch within the dataset. """
        dataset_batch_num = batch_num
        for dataset_num, num_batches in enumerate(self.nums_batches):
            if dataset_batch_num < 0:
                raise ValueError(f"Invalid batch number: {dataset_batch_num}")
            if dataset_batch_num < num_batches:
                return dataset_num, dataset_batch_num
            dataset_batch_num -= num_batches
        raise ValueError(f"Batch {batch_num} is invalid for {self} with "
                         f"{self.num_batches} batch(es)")

    def get_batch(self, batch_num: int):
        # Determine the dataset and the batch number in that dataset.
        dataset_num, dataset_batch_num = self._translate_batch_num(batch_num)
        # Load the batch.
        batch = self.datasets[dataset_num].get_batch(dataset_batch_num)
        # Renumber the batch from the numbering in its original dataset
        # to the numbering in the pooled dataset.
        batch.batch = batch_num
        return batch


class WideDataset(MergedRegionDataset, ABC):
    """ Dataset made by horizontally joining other datasets from one or
    more regions of the same reference sequence. """

    @cached_property
    def datasets(self):
        # Determine the name of the joined region and the individual
        # regions from the report.
        joined_regs = self.report.get_field(JoinedRegionsF)
        # Determine the report file for each joined region.
        load_func = self.get_dataset_load_func()
        region_report_files = [path.cast_path(
            self.report_file,
            self.get_report_type().seg_types(),
            load_func.report_path_seg_types,
            **load_func.report_path_auto_fields,
            reg=reg
        ) for reg in joined_regs]
        if not region_report_files:
            raise ValueError(f"{self} got no datasets")
        return list(map(self.get_dataset_load_func(),
                        region_report_files))

    @cached_property
    def num_batches(self):
        return self._get_common_attr("num_batches")

    @cached_property
    def region_names(self):
        """ Names of all joined regions. """
        return self._list_dataset_attr("region", "name")

    @cached_property
    def region(self):
        return unite(*self._list_dataset_attr("region"),
                     name=self.report.get_field(RegF),
                     refseq=self.refseq)

    @abstractmethod
    def _join(self, batches: Iterable[tuple[str, ReadBatch]]) -> ReadBatch:
        """ Join corresponding batches of data. """

    def get_batch(self, batch_num: int):
        # Join the batch with that number from every dataset.
        return self._join((dataset.region.name, dataset.get_batch(batch_num))
                          for dataset in self.datasets)


class WideMutsDataset(WideDataset, MutsDataset, ABC):
    """ WideDataset with mutation data. """


class MultistepDataset(MutsDataset, ABC):
    """ Dataset made by integrating two datasets from different steps of
    the workflow. """

    @classmethod
    @abstractmethod
    def get_dataset1_load_func(cls) -> LoadFunction:
        """ Function to load Dataset 1. """

    @classmethod
    @abstractmethod
    def get_dataset2_type(cls) -> type[RegionDataset]:
        """ Type of Dataset 2. """

    @classmethod
    def get_dataset2_load_func(cls):
        """ Function to load Dataset 2. """
        return LoadFunction(cls.get_dataset2_type())

    @classmethod
    def get_report_type(cls):
        # The report is for dataset 2.
        return cls.get_dataset2_type().get_report_type()

    @classmethod
    def get_dataset1_report_file(cls, dataset2_report_file: Path):
        """ Given the report file for Dataset 2, determine the report
        file for Dataset 1. """
        load_func = cls.get_dataset1_load_func()
        return path.cast_path(
            dataset2_report_file,
            cls.get_report_type().seg_types(),
            load_func.report_path_seg_types,
            **load_func.report_path_auto_fields
        )

    @classmethod
    def load_dataset1(cls, dataset2_report_file: Path, verify_times: bool):
        """ Load Dataset 1. """
        load_func = cls.get_dataset1_load_func()
        return load_func(cls.get_dataset1_report_file(dataset2_report_file),
                         verify_times=verify_times)

    @classmethod
    def load_dataset2(cls, dataset2_report_file: Path, verify_times: bool):
        """ Load Dataset 2. """
        load_func = cls.get_dataset2_load_func()
        return load_func(dataset2_report_file,
                         verify_times=verify_times)

    def __init__(self, dataset2_report_file: Path, **kwargs):
        super().__init__(dataset2_report_file, **kwargs)
        data1 = self.load_dataset1(dataset2_report_file, self.verify_times)
        data2 = self.load_dataset2(dataset2_report_file, self.verify_times)
        time1 = data1.timestamp
        time2 = data2.timestamp
        if self.verify_times and time1 > time2:
            raise ReversedTimeStampError(
                f"{data2} was presumably made from {data1}, but its report "
                f"file was written earlier: {time2.strftime(DATETIME_FORMAT)} "
                f"compared to {time1.strftime(DATETIME_FORMAT)}. "
                f"If you are sure this inconsistency is not a problem, "
                f"then you can suppress this error using --no-verify-times"
            )
        self.data1 = data1
        self.data2 = data2

    @property
    def refseq(self):
        return self.data1.refseq

    @property
    def timestamp(self):
        return self.data2.timestamp

    @cached_property
    def data_dirs(self):
        return self.data1.data_dirs + self.data2.data_dirs

    @abstractmethod
    def _integrate(self, batch1: MutsBatch, batch2: ReadBatch) -> MutsBatch:
        """ Integrate corresponding batches of data. """

    @property
    def num_batches(self):
        return self.report.get_field(NumBatchF)

    def get_batch(self, batch_num: int):
        return self._integrate(self.data1.get_batch(batch_num),
                               self.data2.get_batch(batch_num))


def load_datasets(input_path: Iterable[str | Path],
                  load_func: LoadFunction,
                  **kwargs):
    """ Yield a Dataset from each report file in `input_path`.

    Parameters
    ----------
    input_path: Iterable[str | Path]
        Input paths to be searched recursively for report files.
    load_func: LoadFunction
        Function to load the dataset from each report file.
    """
    for report_file in path.find_files_chain(input_path,
                                             load_func.report_path_seg_types):
        try:
            yield load_func(report_file, **kwargs)
        except Exception as error:
            logger.error(error)
