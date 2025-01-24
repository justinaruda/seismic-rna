import os
from abc import ABC

import numpy as np
import pandas as pd
from click import command

from .pospair import PositionPairGraph, PositionPairWriter, PositionPairRunner
from ..core.run import log_command

COMMAND = __name__.split(os.path.extsep)[-1]


def calc_phi(n: int | float | np.ndarray | pd.Series,
             a: int | float | np.ndarray | pd.Series | pd.DataFrame,
             b: int | float | np.ndarray | pd.Series | pd.DataFrame,
             a_and_b: int | float | np.ndarray | pd.Series | pd.DataFrame):
    """ Calculate the phi correlation coefficient for a 2x2 matrix.

    +----+----+
    | AB | AO | A.
    +----+----+
    | OB | OO | O.
    +----+----+
      .B   .O   ..

    where
      A. = AB + AO
      .B = AB + OB
      .. = A. + O. = .B + .O

    Parameters
    ----------
    n: int | float | np.ndarray | pd.Series
        Observations in total (..)
    a: int | float | np.ndarray | pd.Series | pd.DataFrame
        Observations for which A is true, regardless of B (A.)
    b: int | float | np.ndarray | pd.Series | pd.DataFrame
        Observations for which B is true, regardless of A (.B)
    a_and_b: int | float | np.ndarray | pd.Series | pd.DataFrame
        Observations for which A and B are both true (AB)

    Returns
    -------
    float | np.ndarray | pd.Series | pd.DataFrame
        Phi correlation coefficient
    """
    a_x_b = a * b
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n * a_and_b - a_x_b) / np.sqrt(a_x_b * (n - a) * (n - b))


class PositionCorrelationGraph(PositionPairGraph, ABC):
    """ Phi correlations between pairs of positions. """

    @classmethod
    def graph_kind(cls):
        return COMMAND

    @classmethod
    def what(cls):
        return "Phi correlation between"

    @classmethod
    def _pair_func(cls):
        """ Function to compare each pair of positions. """
        return calc_phi


class PositionCorrelationWriter(PositionPairWriter):

    @classmethod
    def graph_type(cls):
        return PositionCorrelationGraph


class PositionCorrelationRunner(PositionPairRunner):

    @classmethod
    def get_writer_type(cls):
        return PositionCorrelationWriter

    @classmethod
    @log_command(COMMAND)
    def run(cls, *args, **kwargs):
        return super().run(*args, **kwargs)


@command(COMMAND, params=PositionCorrelationRunner.params())
def cli(*args, **kwargs):
    """ Phi correlations between pairs of positions. """
    return PositionCorrelationRunner.run(*args, **kwargs)

########################################################################
#                                                                      #
# © Copyright 2022-2025, the Rouskin Lab.                              #
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
