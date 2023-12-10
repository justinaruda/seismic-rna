import unittest as ut

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from seismicrna.core.mu.compare import (calc_coeff_determ,
                                        calc_pearson,
                                        calc_rmsd,
                                        calc_spearman)

rng = np.random.default_rng()


class TestCalcPearson(ut.TestCase):

    @classmethod
    def calc_true(cls, x, y):
        """ Calculate the "true" coefficient using a trusted method. """
        return pearsonr(x, y).statistic

    def test_numpy_1d(self):
        # Vary number of rows.
        for nr in range(2, 10):
            x = rng.random(nr)
            y = rng.random(nr)
            s = calc_pearson(x, y)
            self.assertIsInstance(s, float)
            self.assertTrue(np.isclose(s, self.calc_true(x, y)))

    def test_numpy_2d(self):
        # Vary number of columns.
        for nc in range(1, 3):
            # Vary number of rows.
            for nr in range(2, 10):
                x = rng.random((nr, nc))
                y = rng.random((nr, nc))
                s = calc_pearson(x, y)
                self.assertIsInstance(s, np.ndarray)
                self.assertEqual(s.shape, (nc,))
                # Compare the correlation for each column.
                for ic, sc in enumerate(s):
                    self.assertTrue(np.isclose(sc, self.calc_true(x[:, ic],
                                                                  y[:, ic])))

    def test_series(self):
        # Vary number of rows.
        for nr in range(2, 10):
            x = pd.Series(rng.random(nr))
            y = pd.Series(rng.random(nr))
            s = calc_pearson(x, y)
            self.assertIsInstance(s, float)
            self.assertTrue(np.isclose(s, self.calc_true(x, y)))

    def test_dataframe(self):
        # Vary number of columns.
        for nc in range(1, 3):
            # Vary number of rows.
            for nr in range(2, 10):
                x = pd.DataFrame(rng.random((nr, nc)))
                y = pd.DataFrame(rng.random((nr, nc)))
                s = calc_pearson(x, y)
                self.assertIsInstance(s, pd.Series)
                self.assertEqual(s.shape, (nc,))
                # Compare the correlation for each column.
                for ic, sc in enumerate(s):
                    self.assertTrue(np.isclose(
                        sc, self.calc_true(x.iloc[:, ic], y.iloc[:, ic])
                    ))


class TestCalcCoeffDeterm(ut.TestCase):

    @classmethod
    def calc_true(cls, x, y):
        """ Calculate the "true" coefficient using a trusted method. """
        return pearsonr(x, y).statistic ** 2

    def test_dataframe(self):
        # Vary number of columns.
        for nc in range(1, 3):
            # Vary number of rows.
            for nr in range(2, 10):
                x = pd.DataFrame(rng.random((nr, nc)))
                y = pd.DataFrame(rng.random((nr, nc)))
                s = calc_coeff_determ(x, y)
                self.assertIsInstance(s, pd.Series)
                self.assertEqual(s.shape, (nc,))
                # Compare the correlation for each column.
                for ic, sc in enumerate(s):
                    self.assertTrue(np.isclose(
                        sc, self.calc_true(x.iloc[:, ic], y.iloc[:, ic])
                    ))


class TestCalcSpearman(ut.TestCase):

    @classmethod
    def calc_true(cls, x, y):
        """ Calculate the "true" coefficient using a trusted method. """
        return spearmanr(x, y).statistic

    def test_numpy_1d(self):
        # Vary number of rows.
        for nr in range(2, 10):
            x = rng.random(nr)
            y = rng.random(nr)
            s = calc_spearman(x, y)
            self.assertIsInstance(s, float)
            self.assertTrue(np.isclose(s, self.calc_true(x, y)))

    def test_numpy_2d(self):
        # Vary number of columns.
        for nc in range(1, 3):
            # Vary number of rows.
            for nr in range(2, 10):
                x = rng.random((nr, nc))
                y = rng.random((nr, nc))
                s = calc_spearman(x, y)
                self.assertIsInstance(s, np.ndarray)
                self.assertEqual(s.shape, (nc,))
                # Compare the correlation for each column.
                for ic, sc in enumerate(s):
                    self.assertTrue(np.isclose(sc, self.calc_true(x[:, ic],
                                                                  y[:, ic])))

    def test_series(self):
        # Vary number of rows.
        for nr in range(2, 10):
            x = pd.Series(rng.random(nr))
            y = pd.Series(rng.random(nr))
            s = calc_spearman(x, y)
            self.assertIsInstance(s, float)
            self.assertTrue(np.isclose(s, self.calc_true(x, y)))

    def test_dataframe(self):
        # Vary number of columns.
        for nc in range(1, 3):
            # Vary number of rows.
            for nr in range(2, 10):
                x = pd.DataFrame(rng.random((nr, nc)))
                y = pd.DataFrame(rng.random((nr, nc)))
                s = calc_spearman(x, y)
                self.assertIsInstance(s, pd.Series)
                self.assertEqual(s.shape, (nc,))
                # Compare the correlation for each column.
                for ic, sc in enumerate(s):
                    self.assertTrue(np.isclose(
                        sc, self.calc_true(x.iloc[:, ic], y.iloc[:, ic])
                    ))


if __name__ == "__main__":
    ut.main()

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
