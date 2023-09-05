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

import random
import unittest

import numpy as np

from earth_pressure_la import elements as el
from misc import helpers as hel

half_pi = np.pi * 0.5


def coulomb_la(ph, d, a, b, t_1):
    t_v = half_pi + a - t_1 - ph
    v_1 = np.cos(d + a) / np.sin(d + t_1 + ph)
    t_2 = half_pi - a + b
    l_1 = 1 / np.cos(a)
    wedge = el.Wedge(t_1, t_2, l_1, t_v, v_1, v_1)
    return wedge.external_energy(2.)


class TestWedge(unittest.TestCase):

    def test_geometry(self):
        for i in range(10000):
            theta_1 = random.uniform(0., np.pi * .499)
            wedge = el.Wedge(theta_1, .5 * np.pi, 1., 0, 0, 0)
            self.assertTrue(abs(wedge.area() - np.sin(theta_1) * .5))

    def test_rankine_case(self):
        self.assertTrue(abs(hel.kah_coulomb(np.pi / 6, 0, 0, 0) - 1 / 3.) < 1e-6)
        self.assertTrue(abs(coulomb_la(np.pi / 6, 0, 0, 0, np.pi / 6.) - 1 / 3.) < 1e-6)

    def test_wedge(self):
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-40, 41, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))
        counts = [0, 0]
        max_diff = 0
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= beta or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0:
                            continue
                        x = np.linspace(0, min(half_pi + alpha - beta, np.pi - delta * phi - phi) * .99999, 1000)
                        diff = abs(np.max(coulomb_la(phi, delta * phi, alpha, beta, x)) -
                                   hel.kah_coulomb(phi, delta * phi, alpha, beta))
                        if diff > 1e-5:
                            print("phi={:.1f}, delta={:.1f}, alpha={:.1f}, beta={:.1f}: Diff is {:.6e} --> not ok!".
                                  format(np.rad2deg(phi), np.rad2deg(delta * phi), np.rad2deg(alpha),
                                         np.rad2deg(beta), diff))
                            counts[0] += 1
                        else:
                            counts[1] += 1
                        max_diff = max(max_diff, diff)
        print("####\nTest Wedge:")
        print("Total checks passed: {:d}\nTotal checks not passed: {:d}\nMax diff: {:.5e}".format(counts[1], counts[0],
                                                                                                  max_diff))
        print("####")
        self.assertTrue(counts[0] == 0)


if __name__ == '__main__':
    unittest.main()
