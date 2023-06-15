from abc import ABC, abstractmethod
from functools import cache
from itertools import chain
import os
from pathlib import Path

from click import command
import pandas as pd
from plotly import graph_objects as go

from .base import (find_tables, GraphWriter, CartesianGraph, OneTableSeqGraph,
                   OneSampGraph, OneTableSectGraph)
from .color import RelColorMap, SeqColorMap
from ..core import docdef, path
from ..core.cli import (opt_table, opt_fields, opt_stack, opt_count,
                        opt_csv, opt_html, opt_pdf, opt_max_procs, opt_parallel)
from ..core.parallel import dispatch
from ..core.seq import BASES
from ..table.base import CountTable
from ..table.load import (POS_FIELD, TableLoader, RelPosTableLoader,
                          MaskPosTableLoader, ClusterPosTableLoader)

# Number of digits to which to round decimals.
PRECISION = 6

params = [
    opt_table,
    opt_fields,
    opt_count,
    opt_stack,
    opt_csv,
    opt_html,
    opt_pdf,
    opt_max_procs,
    opt_parallel,
]


@command(__name__.split(os.path.extsep)[-1], params=params)
def cli(*args, **kwargs):
    """ Create bar graphs of positional attributes. """
    return run(*args, **kwargs)


@docdef.auto()
def run(table: tuple[str, ...],
        fields: str,
        count: bool,
        stack: bool, *,
        csv: bool,
        html: bool,
        pdf: bool,
        max_procs: int,
        parallel: bool) -> list[Path]:
    """ Run the graph pos module. """
    writers = list(map(PosBarGraphWriter, find_tables(table)))
    return list(chain(*dispatch([writer.write for writer in writers],
                                max_procs, parallel, pass_n_procs=False,
                                kwargs=dict(fields=fields, count=count,
                                            stack=stack, csv=csv,
                                            html=html, pdf=pdf))))


class PosBarGraphWriter(GraphWriter):

    def iter(self, fields: str, count: bool, stack: bool):
        if isinstance(self.table, RelPosTableLoader):
            if stack:
                if count:
                    yield RelCountStackedPosBarGraph(table=self.table,
                                                     codes=fields)
                else:
                    yield RelFracStackedPosBarGraph(table=self.table,
                                                    codes=fields)
            else:
                for field in fields:
                    if count:
                        yield RelCountSerialPosBarGraph(table=self.table,
                                                        codes=field)
                    else:
                        yield RelFracSerialPosBarGraph(table=self.table,
                                                       codes=field)
        elif isinstance(self.table, MaskPosTableLoader):
            if stack:
                if count:
                    yield MaskCountStackedPosBarGraph(table=self.table,
                                                      codes=fields)
                else:
                    yield MaskFracStackedPosBarGraph(table=self.table,
                                                     codes=fields)
            else:
                for field in fields:
                    if count:
                        yield MaskCountSerialPosBarGraph(table=self.table,
                                                         codes=field)
                    else:
                        yield MaskFracSerialPosBarGraph(table=self.table,
                                                        codes=field)
        elif isinstance(self.table, ClusterPosTableLoader):
            for cluster in self.table.cluster_names:
                yield ClustPosBarGraph(table=self.table, cluster=cluster)


class PosBarGraph(CartesianGraph, OneTableSeqGraph, OneSampGraph, ABC):
    """ Bar graph wherein each bar represents one sequence position. """

    def __init__(self, *args,
                 table: (RelPosTableLoader
                         | MaskPosTableLoader
                         | ClusterPosTableLoader),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.table = table

    @classmethod
    @abstractmethod
    def get_source(cls):
        """ Step from which the data came. """
        return ""

    @classmethod
    def get_table_type(cls):
        return RelPosTableLoader, MaskPosTableLoader, ClusterPosTableLoader

    @classmethod
    def get_data_type(cls):
        return pd.Series

    @classmethod
    def get_cmap_type(cls):
        return SeqColorMap

    @classmethod
    def get_xattr(cls):
        return POS_FIELD


class SerialPosBarGraph(PosBarGraph, ABC):
    """ Bar graph with a single series of data. """

    def get_traces(self):
        traces = list()
        # Construct a trace for each type of base.
        for base in BASES:
            # Find the non-missing value at every base of that type.
            vals = self.data.loc[self.seqarr == base].dropna()
            # Check if there are any values to graph.
            if vals.size > 0:
                # Define the text shown on hovering over a bar.
                hovertext = [f"{chr(base)}{x}: {y}" for x, y in vals.items()]
                # Create a trace comprising all bars for this base type.
                traces.append(go.Bar(name=chr(base), x=vals.index, y=vals,
                                     marker_color=self.cmap[base],
                                     hovertext=hovertext,
                                     hoverinfo="text"))
        return traces


class FieldPosBarGraph(PosBarGraph, ABC):
    """ Bar graph of a table with multiple types of bit fields, with one
    bar for each position in the sequence. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._codes = None

    @property
    def codes(self):
        """ One-letter codes of the fields to graph. """
        if self._codes is None:
            raise TypeError("codes not set")
        return self._codes

    @classmethod
    @abstractmethod
    def _validate_codes(cls, codes: str):
        """ Confirm that the codes are valid. """

    @codes.setter
    def codes(self, codes: str):
        self._validate_codes(codes)
        self._codes = codes

    @classmethod
    @abstractmethod
    def is_stacked(cls):
        """ Whether each bar is a stack of fields. """
        return False

    @classmethod
    @abstractmethod
    def y_counts(cls):
        """ Whether the y-axis represents counts. """
        return False

    @classmethod
    def get_yattr(cls):
        return "Count" if cls.y_counts() else "Fraction"

    @classmethod
    def get_table_field(cls, table: CountTable | TableLoader, code: str):
        """ Load one field from the table. """
        return (table.get_field_count(code) if cls.y_counts()
                else table.get_field_frac(code).round(PRECISION))

    @property
    def title(self):
        fields = '/'.join(sorted(CountTable.FIELD_CODES[c] for c in self.codes))
        return (f"{self.sample}: {self.get_source()} {self.get_yattr()}s of "
                f"{fields} bases in {self.ref}")

    @property
    def graph_filename(self):
        sort_codes = "".join(sorted(self.codes))
        by = "stacked" if self.is_stacked() else "serial"
        fname = f"{self.get_source()}_{self.get_yattr()}_{sort_codes}_{by}"
        return fname.lower()


class CountFieldPosBarGraph(FieldPosBarGraph, ABC):
    """ FieldPosBarGraph where each bar represents a count of reads. """

    @classmethod
    def y_counts(cls):
        return True


class FracFieldPosBarGraph(FieldPosBarGraph, ABC):
    """ FieldPosBarGraph where each bar represents a fraction of reads. """

    @classmethod
    def y_counts(cls):
        return False


class SerialFieldPosBarGraph(FieldPosBarGraph, SerialPosBarGraph, ABC):
    """ Bar graph wherein each bar represents a base in a sequence. """

    def __init__(self, *args, codes: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.codes = codes

    @classmethod
    def is_stacked(cls):
        return False

    @classmethod
    def _validate_codes(cls, codes: str):
        if len(codes) != 1:
            raise ValueError(f"Expected 1 field code, but got {len(codes)}")

    @property
    def code(self):
        """ The code of the field to graph. """
        return self.codes[0]

    def _get_data(self) -> pd.Series:
        return self.get_table_field(self.table, self.code)


class StackedFieldPosBarGraph(FieldPosBarGraph, ABC):
    """ Stacked bar graph wherein each stacked bar represents multiple
    outcomes for a base in a sequence. """

    def __init__(self, *args, codes: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.codes = codes

    @classmethod
    def is_stacked(cls):
        return True

    @classmethod
    def get_cmap_type(cls):
        return RelColorMap

    @classmethod
    def get_data_type(cls):
        return pd.DataFrame

    @classmethod
    def _validate_codes(cls, codes: str):
        if len(codes) == 0:
            raise ValueError("Expected 1 or more field codes, but got 0")

    def _get_data(self):
        data = dict()
        for code in self.codes:
            series = self.get_table_field(self.table, code)
            data[series.name] = series
        return pd.DataFrame.from_dict(data)

    def get_traces(self):
        traces = list()
        # Construct a trace for each field.
        for field, vals in self.data.items():
            # Define the text shown on hovering over a bar.
            hovertext = [f"{chr(base)}{x} {field}: {y}"
                         for base, (x, y) in zip(self.seq, vals.items(),
                                                 strict=True)]
            # Create a trace comprising all bars for this field.
            traces.append(go.Bar(name=field, x=vals.index, y=vals,
                                 marker_color=self.cmap[field],
                                 hovertext=hovertext,
                                 hoverinfo="text"))
        return traces

    @cache
    def get_figure(self):
        fig = super().get_figure()
        # Stack the bars at each position.
        fig.update_layout(barmode="stack")
        return fig


class SectPosBarGraph(PosBarGraph, OneTableSectGraph, ABC):
    """ Bar graph of the positions in a section. """

    @property
    def title(self):
        return f"{super().title}:{self.sect}"


class RelPosBarGraph(FieldPosBarGraph, ABC):
    """ Bar graph of related data from one sample at each position in a
    sequence. """

    @classmethod
    def get_source(cls):
        return "Related"


class MaskPosBarGraph(FieldPosBarGraph, SectPosBarGraph, ABC):
    """ Bar graph of masked data from one sample at each position in a
    sequence. """

    @classmethod
    def get_source(cls):
        return "Masked"


class RelFracSerialPosBarGraph(SerialFieldPosBarGraph,
                               FracFieldPosBarGraph,
                               RelPosBarGraph):
    """ """


class RelFracStackedPosBarGraph(StackedFieldPosBarGraph,
                                FracFieldPosBarGraph,
                                RelPosBarGraph):
    """ """


class RelCountSerialPosBarGraph(SerialFieldPosBarGraph,
                                CountFieldPosBarGraph,
                                RelPosBarGraph):
    """ """


class RelCountStackedPosBarGraph(StackedFieldPosBarGraph,
                                 CountFieldPosBarGraph,
                                 RelPosBarGraph):
    """ """


class MaskFracSerialPosBarGraph(SerialFieldPosBarGraph,
                                FracFieldPosBarGraph,
                                MaskPosBarGraph):
    """ """


class MaskFracStackedPosBarGraph(StackedFieldPosBarGraph,
                                 FracFieldPosBarGraph,
                                 MaskPosBarGraph):
    """ """


class MaskCountSerialPosBarGraph(SerialFieldPosBarGraph,
                                 CountFieldPosBarGraph,
                                 MaskPosBarGraph):
    """ """


class MaskCountStackedPosBarGraph(StackedFieldPosBarGraph,
                                  CountFieldPosBarGraph,
                                  MaskPosBarGraph):
    """ """


class ClustPosBarGraph(SerialPosBarGraph, SectPosBarGraph):
    """ Bar graph of a table of per-cluster mutation rates. """

    def __init__(self, *args, cluster: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster = cluster

    @classmethod
    def get_source(cls):
        return "Clustered"

    @classmethod
    def get_yattr(cls):
        return "Mutation Rate"

    @property
    def title(self):
        return (f"{self.sample}: {self.get_source()} {self.get_yattr()}s of "
                f"bases in {self.ref}:{self.sect}, {self.cluster}")

    def _get_data(self):
        return self.table.data[self.cluster]

    @property
    def graph_filename(self):
        return path.fill_whitespace(self.cluster).lower()
