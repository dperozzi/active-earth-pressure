#  Copyright (c) 2023. ETH Zurich, David Perozzi; D-BAUG; Institute for Geotechnical Engineering; Chair of Geomechanics and Geosystems Engineering
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np

from earth_pressure_la import mechanisms as mec
from earth_pressure_la import static_solution as sts
from misc import helpers as hel

half_pi = np.pi * 0.5


class TestStaticSolution(unittest.TestCase):
    def test_coulomb_case(self):
        """
        Test the static solution for the Coulomb case, i.e. for the case with arbitrary wall friction, but in which the
        stress state does not need to be rotated (i.e. theta=0). In that case, the failure surface is linear and Coulomb
        corresponds to the exact solution.
        """
        for i in range(1000):
            beta = np.deg2rad(np.random.uniform(-45, 45))
            phi = np.random.uniform(1., abs(beta))
            delta = np.random.uniform(0, abs(phi))
            alpha = 0.5 * (
                    -np.arcsin(np.sin(beta) / np.sin(phi)) + beta + np.arcsin(np.sin(delta) / np.sin(phi)) - delta)
            static_sol = sts.LancellottaExtended()
            static_sol.set_parameters(phi, delta, alpha, beta, 1)
            static_sol.compute()
            one_wedge_t = mec.OneWedge("T")
            one_wedge_t.set_parameters(phi, delta, alpha, beta, 1)
            one_wedge_t.optimize()
            one_wedge_rf = mec.OneWedge("RF")
            one_wedge_rf.set_parameters(phi, delta, alpha, beta, 1)
            one_wedge_rf.optimize()
            kah = hel.kah_coulomb(phi, delta, alpha, beta)
            ka = hel.ka_coulomb(phi, delta, alpha, beta)
            kan = hel.kan_coulomb(phi, delta, alpha, beta)
            eah = hel.eah_coulomb(1, 1, phi, delta, alpha, beta)
            ea = hel.ea_coulomb(1, 1, phi, delta, alpha, beta)
            ma = hel.ma_coulomb(1, 1, phi, delta, alpha, beta)
            self.assertLess(abs(static_sol.k_h_norm[0] - one_wedge_t.kah) / static_sol.k_h_norm[0], 1e-4)
            self.assertLess(abs(static_sol.k_res_norm[0] - one_wedge_t.ka) / static_sol.k_res_norm[0], 1e-4)
            self.assertLess(abs(static_sol.k_n_norm[0] - one_wedge_t.kan) / static_sol.k_n_norm[0], 1e-4)
            self.assertLess(abs(static_sol.k_h_norm[0] - one_wedge_rf.kah) / static_sol.k_h_norm[0], 1e-4)
            self.assertLess(abs(static_sol.k_res_norm[0] - one_wedge_rf.ka) / static_sol.k_res_norm[0], 1e-4)
            self.assertLess(abs(static_sol.k_n_norm[0] - one_wedge_rf.kan) / static_sol.k_n_norm[0], 1e-4)
            self.assertLess(
                abs(one_wedge_t.resultant_force - static_sol.earthpressure_h[0]) / one_wedge_t.resultant_force, 1e-4)
            self.assertLess(
                abs(one_wedge_rf.resultant_force - static_sol.moment_wallbase[0]) / one_wedge_rf.resultant_force, 1e-4)
            self.assertLess(abs(kan - static_sol.k_n_norm[0]) / kan, 1e-4)
            self.assertLess(abs(kah - static_sol.k_h_norm[0]) / kah, 1e-4)
            self.assertLess(abs(ka - static_sol.k_res_norm[0]) / ka, 1e-4)
            self.assertLess(abs(eah - static_sol.earthpressure_h[0]) / eah, 1e-4)
            self.assertLess(abs(ea - static_sol.earthpressure_res[0]) / ea, 1e-4)
            self.assertLess(abs(ma - static_sol.moment_wallbase[0]) / ma, 1e-4)
