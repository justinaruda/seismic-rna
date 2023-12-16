import os
from functools import cached_property, partial
from logging import getLogger

from click import command
from plotly import graph_objects as go

from .twotable import TwoTableMergedGraph, TwoTableRunner, TwoTableWriter
from .traces import iter_seq_line_traces
from ..core.arg import opt_mucomp, opt_window, opt_winmin
from ..core.mu import compare_windows, get_comp_name

logger = getLogger(__name__)

COMMAND = __name__.split(os.path.extsep)[-1]


class RollingCorrelationGraph(TwoTableMergedGraph):

    @classmethod
    def graph_kind(cls):
        return COMMAND

    @classmethod
    def what(cls):
        return "Rolling correlation"

    @classmethod
    def _trace_function(cls):
        return iter_seq_line_traces

    def __init__(self, *, mucomp: str, window: int, winmin: int, **kwargs):
        super().__init__(**kwargs)
        self._method = mucomp
        self._size = window
        self._min_count = winmin

    @cached_property
    def predicate(self):
        return "_".join(
            [super().predicate,
             "-".join(map(str, [self._method, self._size, self._min_count]))]
        )

    @cached_property
    def details(self):
        return super().details + [f"metric = {self._method.upper()}",
                                  f"window = {self._size} nt",
                                  f"min = {self._min_count} nt"]

    @cached_property
    def y_title(self):
        return f"{get_comp_name(self._method)} of {self.data_kind}s"

    @cached_property
    def _merge_data(self):
        return partial(compare_windows,
                       method=self._method,
                       size=self._size,
                       min_count=self._min_count)

    def _figure_layout(self, fig: go.Figure):
        super()._figure_layout(fig)
        fig.update_yaxes(gridcolor="#d0d0d0")


class RollingCorrelationWriter(TwoTableWriter):

    @classmethod
    def get_graph_type(cls):
        return RollingCorrelationGraph


class RollingCorrelationRunner(TwoTableRunner):

    @classmethod
    def var_params(cls):
        return super().var_params() + [opt_mucomp, opt_window, opt_winmin]

    @classmethod
    def writer_type(cls):
        return RollingCorrelationWriter


@command(RollingCorrelationGraph.graph_kind(),
         params=RollingCorrelationRunner.params())
def cli(*args, **kwargs):
    """ Create line graphs of rolling correlations between datasets. """
    return RollingCorrelationRunner.run(*args, **kwargs)

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
