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
from misc import helpers as hel
from misc.definitions import ALPHA, BETA, GAMMA, DELTA, PHI

half_pi = np.pi * 0.5

rel_tol = 1e-4


class TestOneWedge(unittest.TestCase):
    """
    TestOneWedge is a subclass of unittest.TestCase that contains several test methods for testing the functionality
    of the OneWedge class.

    Methods:
    - test_rankine_case: This method tests the most basic case where phi=30deg, alpha=0, beta=0, delta=0. It checks if
    the resultant earth pressure calculated by the OneWedge class matches the expected value.
    - test_rankine_case_optimization: This method tests the optimization function of the OneWedge class. It sets the
    parameters to phi=30deg, alpha=0, beta=0, delta=0 and then optimizes the angle. It checks if the optimized resultant earth pressure matches the expected value and if the optimized angle matches the expected angle.
    - test_rankine_case_rotation: Similar to test_rankine_case_optimization, but tests the rotation about the wall foot.
    - test_general_bc_translation: This method tests a broader range of parameters against Coulomb's solution
    considering translation. It iterates over different values of phi, delta, alpha, and beta, sets the parameters to each combination, optimizes the angle, and compares the resultant earth pressure with Coulomb's solution.
    - test_general_bc_rotation_foot: Similar to test_general_bc_translation, but tests the rotation about the wall foot.

    """

    def test_rankine_case(self):
        # The most basic test: phi=30deg, alpha=0, beta=0, delta=0
        # According to Rankine's and Coulomb's theories, the resultant earth pressure should be 1/3
        one_wedge = mec.OneWedge("T")
        one_wedge.set_parameters(np.pi / 6., 0, 0, 0)
        ea = -one_wedge._external_energy(np.array([np.pi / 6.]))
        true_sol = 1. / 3.
        self.assertLess(abs(ea - true_sol) / true_sol, 1e-5)

    def test_rankine_case_optimization(self):
        # The same as before, but this time we also test the optimization function
        # the optimal configuration should correspond to 45deg-phi/2=30deg
        one_wedge = mec.OneWedge("T")
        one_wedge.set_parameters(np.pi / 6., 0, 0, 0)
        one_wedge.optimize([np.deg2rad(10)])
        result = -one_wedge.optimize_result.fun
        x_opt = one_wedge.optimize_result.x[0]
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        true_sol_angle = np.pi / 6.
        self.assertLess(abs(x_opt - true_sol_angle) / true_sol_angle, rel_tol)

    def test_rankine_case_rotation(self):
        # Again the same, but this time we consider rotation about the wall foot
        one_wedge = mec.OneWedge("RF")
        one_wedge.set_parameters(np.pi / 6., 0, 0, 0)
        one_wedge.optimize([np.deg2rad(40)])
        result = -one_wedge.optimize_result.fun
        x_opt = one_wedge.optimize_result.x[0]
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        true_sol_angle = np.pi / 6.
        self.assertLess(abs(x_opt - true_sol_angle) / true_sol_angle, rel_tol)

    def test_general_bc_translation(self):
        # Test a broader range of parameters against Coulomb's solution considering translation
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-40, 41, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("T")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= np.abs(beta) or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        coulomb_sol = hel.eah_coulomb(one_wedge.params[GAMMA], one_wedge.h_soil, phi, delta * phi,
                                                      alpha, beta)
                        diff = abs(one_wedge.resultant_force - coulomb_sol) / coulomb_sol
                        if abs(one_wedge.resultant_force) < 1e-4 and abs(coulomb_sol) < 1e-4:
                            continue
                        if diff >= rel_tol:
                            print(
                                "phi={:.1f}, delta={:.1f}, alpha={:.1f}, beta={:.1f}: Diff is {:.6e} --> not ok!".format(
                                    np.rad2deg(phi), np.rad2deg(delta * phi), np.rad2deg(alpha), np.rad2deg(beta),
                                    diff))
                        self.assertLess(diff, rel_tol)
                        kah_coulomb = hel.kah_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.kah - kah_coulomb) / kah_coulomb, rel_tol)
                        ka_coulomb = hel.ka_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.ka - ka_coulomb) / ka_coulomb, rel_tol)
                        kan_coulomb = hel.kan_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.kan - kan_coulomb) / kan_coulomb, rel_tol)

    def test_general_bc_rotation_foot(self):
        # Test a broader range of parameters against Coulomb's solution considering rotation about the wall foot
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-20, 21, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("RF")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= np.abs(beta) or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0:
                            continue
                        theta = phi + np.arctan(np.cos(phi - alpha) / (np.sin(phi - alpha) + np.sqrt(
                            np.sin(phi + delta) * np.cos(-beta + alpha) / (
                                    np.sin(phi - beta) * np.cos(alpha + delta)))))
                        angle = half_pi + alpha - theta
                        if half_pi - angle - phi < 0:
                            continue
                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        coulomb_sol = hel.ma_coulomb(one_wedge.params[GAMMA], one_wedge.h_soil, phi, delta * phi, alpha,
                                                     beta)
                        diff = abs(one_wedge.resultant_force - coulomb_sol) / coulomb_sol

                        if diff >= 1e-5:
                            print("phi={:.1f}, delta={:.1f}, alpha={:.1f}, beta={:.1f}: Diff is {:.6e} --> not ok!".
                                  format(np.rad2deg(phi), np.rad2deg(delta * phi), np.rad2deg(alpha),
                                         np.rad2deg(beta), diff))
                            print("half_pi-t_11-phi={:.5e}".format(half_pi - one_wedge.optimized_config[0] - phi))
                            theta = one_wedge.params[PHI] + np.arctan(
                                np.cos(one_wedge.params[PHI] - one_wedge.params[ALPHA]) / (
                                        np.sin(one_wedge.params[PHI] - one_wedge.params[ALPHA]) + np.sqrt(
                                    np.sin(one_wedge.params[PHI] + one_wedge.params[DELTA]) * np.cos(
                                        -one_wedge.params[BETA] + one_wedge.params[ALPHA]) / (
                                            np.sin(one_wedge.params[PHI] - one_wedge.params[BETA]) * np.cos(
                                        one_wedge.params[ALPHA] + one_wedge.params[DELTA])))))
                            angle = half_pi + one_wedge.params[ALPHA] - theta
                            print("half_pi-t_coul-phi={:.5e}".format(half_pi - angle - phi))

                        self.assertLess(diff, rel_tol)
                        kah_coulomb = hel.kah_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.kah - kah_coulomb) / kah_coulomb, rel_tol)
                        ka_coulomb = hel.ka_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.ka - ka_coulomb) / ka_coulomb, rel_tol)
                        kan_coulomb = hel.kan_coulomb(phi, delta * phi, alpha, beta)
                        self.assertLess(abs(one_wedge.kan - kan_coulomb) / kan_coulomb, rel_tol)


class TestTwoWedges(unittest.TestCase):
    """

    This class is used to test the functionality of the TwoWedges class from the mechanisms module.

    Methods
    -------
    - test_rankine_case: Tests the calculation of earth pressure for a basic case using Rankine's theory.
    - test_rankine_case_optimization: Tests the optimization function with Rankine's theory.
    - test_rankine_case_rotation: Tests the optimization function with rotation about the wall foot.
    - test_general_bc_translation: Tests a broader range of parameters against Coulomb's solution considering translation.
    - test_general_bc_rotation_foot: Tests a broader range of parameters against Coulomb's solution considering rotation about the wall foot.

    """

    def test_rankine_case(self):
        # The most basic test: phi=30deg, alpha=0, beta=0, delta=0
        # According to Rankine's and Coulomb's theories, the resultant earth pressure should be 1/3
        mech = mec.TwoWedges("T")
        mech.set_parameters(np.pi / 6., 0, 0, 0)
        t_12 = np.linspace(0, half_pi, 50)
        for x in t_12:
            ea = -mech._external_energy(np.array([np.pi / 6., x, half_pi + x - np.pi / 3.]))
            true_sol = 1. / 3.
            self.assertLess(abs(ea - true_sol) / true_sol, 1e-5)

    def test_rankine_case_optimization(self):
        # The same as before, but this time we also test the optimization function
        # the optimal configuration should correspond to 45deg-phi/2=30deg
        mech = mec.TwoWedges("T")
        mech.set_parameters(np.pi / 6., 0, 0, 0)
        mech.optimize()
        result = -mech.optimize_result.fun
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        true_sol_angle = np.pi / 6.
        self.assertLess(abs(mech.optimize_result.x[0] - true_sol_angle) / true_sol_angle, 3 * rel_tol)
        self.assertLess(
            abs(half_pi + mech.optimize_result.x[1] - mech.optimize_result.x[2] - np.deg2rad(60)) / np.deg2rad(
                60), 3 * rel_tol)

    def test_rankine_case_rotation(self):
        # Again the same, but this time we consider rotation about the wall foot
        mech = mec.TwoWedges("RF")
        mech.set_parameters(np.pi / 6., 0, 0, 0)
        mech.optimize()
        result = -mech.optimize_result.fun
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        true_sol_angle = np.pi / 6.
        self.assertLess(abs(mech.optimize_result.x[0] - true_sol_angle) / true_sol_angle, 3 * rel_tol)
        self.assertLess(
            abs(half_pi + mech.optimize_result.x[1] - mech.optimize_result.x[2] - np.deg2rad(60)) / np.deg2rad(
                60), 3 * rel_tol)

    def test_general_bc_translation(self):
        # Test a broader range of parameters against Coulomb's solution considering translation
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-40, 41, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("T")
        two_wedges = mec.TwoWedges("T")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= beta or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(
                                                                                      -alpha + beta)) < 0 or \
                                half_pi + alpha <= phi:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        two_wedges.set_parameters(phi, delta * phi, alpha, beta)
                        t_11 = one_wedge.optimized_config[0]
                        t_12 = np.random.uniform(0, half_pi - alpha + beta)
                        t_21 = t_12 + t_11
                        factor = two_wedges.params[GAMMA] * two_wedges.h_soil ** 2. / 2.
                        kah = -two_wedges._external_energy(np.array([t_11, t_12, t_21])) / factor
                        diff = abs(one_wedge.kah - kah) / kah
                        self.assertLess(diff, rel_tol)
                        two_wedges.optimize()
                        self.assertLess(one_wedge.kah - two_wedges.kah, 1e-5)

    def test_general_bc_rotation_foot(self):
        # Test a broader range of parameters against Coulomb's solution considering rotation about the wall foot
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-20, 21, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("RF")
        two_wedges = mec.TwoWedges("RF")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= beta or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0 or \
                                half_pi + alpha <= phi:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        two_wedges.set_parameters(phi, delta * phi, alpha, beta)
                        t_11 = one_wedge.optimized_config[0]
                        t_12 = np.random.uniform(0, half_pi - alpha + beta)
                        t_21 = t_12 + t_11
                        factor = two_wedges.params[GAMMA] * two_wedges.h_soil ** 3. / (
                                6. * np.cos(alpha) * np.cos(alpha + delta * phi)) * np.cos(delta * phi)
                        kah = -two_wedges._external_energy(np.array([t_11, t_12, t_21])) / factor
                        diff = abs(one_wedge.kah - kah) / kah
                        self.assertLess(diff, rel_tol)
                        two_wedges.optimize()
                        self.assertLess(one_wedge.kah - two_wedges.kah, 1e-4)


class TestExtendedOneWedge(unittest.TestCase):
    """
    This class is used for testing the functionality of the `ExtendedOneWedge` class in the `mechanisms` module.

    Methods:
    - test_coulomb_case_rotation: This method tests the basic functionality of the `ExtendedOneWedge` class assuming rotational movement of the wall. It compares the solution with that of the `OneWedge` class with the same parameters.
    - test_coulomb_case_translation: This method tests the basic functionality of the `ExtendedOneWedge` class assuming translational movement of the wall. It compares the solution with that of the `OneWedge` class with the same parameters.
    - test_general_bc_rotation_foot: This method tests a broader range of parameters against Coulomb's solution considering rotation about the wall foot. It iterates over different combinations of parameters and compares the solution with that of the `OneWedge` class with the same parameters.

    """

    def test_coulomb_case_rotation(self):
        # The most basic test assuming a wall rotation: phi=30deg, alpha=0, beta=0, delta=0; later delta=phi
        # the solution is then compared to that of the single wedge mechanism
        ext_onewedge = mec.ExtendedOneWedge("RF")
        ext_onewedge.set_parameters(np.pi / 6., 0, 0, 0)
        one_wedge = mec.OneWedge("RF")
        one_wedge.set_parameters(np.pi / 6., 0., 0, 0)
        one_wedge.optimize()
        ea = -ext_onewedge._external_energy(np.array([0., one_wedge.optimize_result.x[0]]))
        self.assertLess(abs(ea - one_wedge.resultant_force) / one_wedge.resultant_force, rel_tol)
        ext_onewedge = mec.ExtendedOneWedge("RF")
        ext_onewedge.set_parameters(np.pi / 6., np.pi / 6., 0, 0)
        one_wedge = mec.OneWedge("RF")
        one_wedge.set_parameters(np.pi / 6., np.pi / 6., 0, 0)
        one_wedge.optimize()
        ea = -ext_onewedge._external_energy(np.array([0., one_wedge.optimize_result.x[0]]))
        self.assertLess(abs(ea - one_wedge.resultant_force) / one_wedge.resultant_force, rel_tol)

    def test_coulomb_case_translation(self):
        # The most basic test assuming a translation: phi=30deg, alpha=0, beta=0, delta=0; later delta=phi
        # the solution is then compared to that of the single wedge mechanism
        ext_onewedge = mec.ExtendedOneWedge("T")
        ext_onewedge.set_parameters(np.pi / 6., 0, 0, 0)
        one_wedge = mec.OneWedge("T")
        one_wedge.set_parameters(np.pi / 6., 0., 0, 0)
        one_wedge.optimize()
        ea = -ext_onewedge._external_energy(np.array([0., one_wedge.optimize_result.x[0]]))
        self.assertLess(abs(ea - one_wedge.resultant_force) / one_wedge.resultant_force, rel_tol)
        ext_onewedge = mec.ExtendedOneWedge("T")
        ext_onewedge.set_parameters(np.pi / 6., np.pi / 6., 0, 0)
        one_wedge = mec.OneWedge("T")
        one_wedge.set_parameters(np.pi / 6., np.pi / 6., 0, 0)
        one_wedge.optimize()
        ea = -ext_onewedge._external_energy(np.array([0., one_wedge.optimize_result.x[0]]))
        self.assertLess(abs(ea - one_wedge.resultant_force) / one_wedge.resultant_force, rel_tol)

    def test_general_bc_rotation_foot(self):
        # Test a broader range of parameters against Coulomb's solution considering rotation about the wall foot
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [1]
        alpha_range = np.deg2rad(range(-30, 41, 8))
        beta_range = np.deg2rad(range(0, 41, 5))

        one_wedge = mec.OneWedge("RF")
        ext_onewedge = mec.ExtendedOneWedge("RF")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= np.abs(beta) or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        ext_onewedge.set_parameters(phi, delta * phi, alpha, beta)
                        diff = abs(one_wedge.resultant_force + ext_onewedge._external_energy(
                            np.array([0., one_wedge.optimize_result.x[0]]))) / one_wedge.resultant_force
                        self.assertLess(diff, rel_tol)
                        ext_onewedge.optimize()
                        diff = one_wedge.resultant_force - ext_onewedge.resultant_force
                        self.assertLess(diff, 5e-4)


class TestLogSpiral(unittest.TestCase):
    """
    This class is used to test the functionality of the TestLogSpiral class from the mechanisms module.

    Methods:
    - test_rankine_case: Tests the basic case where phi=30deg, alpha=0, beta=0, delta=0 and checks if the resultant earth pressure is 1/3.
    - test_rankine_case_optimization: Tests the basic case with optimization where phi=30deg, alpha=0, beta=0, delta=0 and checks if the resultant earth pressure is 1/3 and returns the optimal configuration.
    - test_rankine_case_rotation: Tests the basic case with rotation about the wall foot where phi=30deg, alpha=0, beta=0, delta=0 and checks if the resultant earth pressure is 1/3 and returns the optimal configuration.
    - test_general_bc_translation: Tests a broader range of parameters against Coulomb's solution considering translation.
    - test_general_bc_rotation_foot: Tests a broader range of parameters against Coulomb's solution considering rotation about the wall foot.
    """

    def test_rankine_case(self):
        # The most basic test: phi=30deg, alpha=0, beta=0, delta=0
        # According to Rankine's and Coulomb's theories, the resultant earth pressure should be 1/3
        mech = mec.LogSpiral("T")
        phi = np.pi / 6.
        mech.set_parameters(phi, 0, 0, 0)
        ea = -mech._external_energy(np.array([0.5 * (half_pi - phi), 0.]))
        true_sol = 1. / 3.
        self.assertLess(abs(ea - true_sol) / true_sol, 1e-5)

    def test_rankine_case_optimization(self):
        # The same as before, but this time we also test the optimization function
        # the optimal configuration should correspond to 45deg-phi/2=30deg
        mech = mec.LogSpiral("T")
        phi = np.pi / 6.
        mech.set_parameters(phi, 0, 0, 0)
        mech.optimize()
        result = -mech.optimize_result.fun
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        self.assertLess(abs(mech.optimize_result.x[0] - 0.5 * (half_pi - phi)) / (0.5 * (half_pi - phi)), 1e-3)
        self.assertLess(abs(mech.optimize_result.x[1]), 1e-3)

    def test_rankine_case_rotation(self):
        # Again the same, but this time we consider rotation about the wall foot
        mech = mec.LogSpiral("RF")
        phi = np.pi / 6.
        mech.set_parameters(phi, 0, 0, 0)
        mech.optimize()
        result = -mech.optimize_result.fun
        true_sol = 1. / 3.
        self.assertLess(abs(result - true_sol) / true_sol, 1e-5)
        self.assertLess(abs(mech.optimize_result.x[0] - 0.5 * (half_pi - phi)) / (0.5 * (half_pi - phi)), 1e-3)
        self.assertLess(abs(mech.optimize_result.x[1]), 1e-3)

    def test_general_bc_translation(self):
        # Test a broader range of parameters against Coulomb's solution considering translation
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-40, 41, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("T")
        log_spiral = mec.LogSpiral("T")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= beta or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(
                                                                                      -alpha + beta)) < 0 or \
                                half_pi + alpha <= phi:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        log_spiral.set_parameters(phi, delta * phi, alpha, beta)
                        log_spiral.optimize()

                        optimum_one_wedge = np.array(
                            [half_pi - one_wedge.optimize_result.x[0] - log_spiral.params[PHI], 0])
                        admissible_pt = log_spiral._check_initial_value(np.copy(optimum_one_wedge),
                                                                        log_spiral.bounds[0], [log_spiral.lconstr])
                        if np.allclose(admissible_pt, optimum_one_wedge):
                            t_11 = half_pi - one_wedge.optimized_config[0] - phi
                            factor = log_spiral.params[GAMMA] * log_spiral.h_soil ** 2. / 2.
                            kah = -log_spiral._external_energy(np.array([t_11, 0.])) / factor
                            self.assertLess(abs(one_wedge.kah - kah) / kah, rel_tol)
                            if one_wedge.kah - log_spiral.kah > 1e-5:  # in case the assumption of a logspiral is too
                                # restrictive, the one-wedge mechanism may be governing
                                # In that case, check at least that the logspiral mechanism can give the same solution
                                # as the one-wedge
                                log_spiral.optimize(
                                    np.array([half_pi - one_wedge.optimize_result.x[0] - log_spiral.params[PHI], 0]))
                                self.assertLess(abs(one_wedge.optimize_result.fun - log_spiral._external_energy(
                                    optimum_one_wedge)) / one_wedge.optimize_result.fun, rel_tol)
                            else:  # Generally, the logspiral mechanism should give a higher limit load than the one wedge
                                self.assertLess(one_wedge.kah - log_spiral.kah, 1e-5)
                        else:
                            print(
                                "The optimal configuration of the OneWedge mech. lies outside the admissible configurations for the LogSandwich. No test can be performed.")
                            print("K_ah{{1Wedge}} = {:.5f} K_ah{{LogSandwich}} = {:.5f}".format(one_wedge.kah,
                                                                                                log_spiral.kah))

    def test_general_bc_rotation_foot(self):
        # Test a broader range of parameters against Coulomb's solution considering rotation about the wall foot
        phi_range = np.deg2rad(range(5, 51, 5))
        delta_factor_range = [x * .1 for x in range(11)]
        alpha_range = np.deg2rad(range(-20, 21, 10))
        beta_range = np.deg2rad(range(-40, 41, 10))

        one_wedge = mec.OneWedge("RF")
        log_spiral = mec.LogSpiral("RF")
        for phi in phi_range:
            for delta in delta_factor_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        if (beta - alpha) <= -half_pi or beta - alpha >= half_pi or delta * phi + alpha == half_pi or \
                                -alpha + beta == half_pi or phi <= beta or \
                                np.sin(phi + delta * phi) * np.sin(phi - beta) / (np.cos(delta * phi + alpha) *
                                                                                  np.cos(-alpha + beta)) < 0 or \
                                half_pi + alpha <= phi:
                            continue

                        one_wedge.set_parameters(phi, delta * phi, alpha, beta)
                        one_wedge.optimize()
                        log_spiral.set_parameters(phi, delta * phi, alpha, beta)
                        log_spiral.optimize()

                        optimum_one_wedge = np.array(
                            [half_pi - one_wedge.optimize_result.x[0] - log_spiral.params[PHI], 0])
                        admissible_pt = log_spiral._check_initial_value(np.copy(optimum_one_wedge),
                                                                        log_spiral.bounds[0], [log_spiral.lconstr])
                        if np.allclose(admissible_pt, optimum_one_wedge):
                            t_11 = half_pi - one_wedge.optimized_config[0] - phi
                            factor = log_spiral.params[GAMMA] * log_spiral.h_soil ** 3. / (
                                    6. * np.cos(alpha) * np.cos(alpha + delta * phi)) * np.cos(delta * phi)
                            kah = -log_spiral._external_energy(np.array([t_11, 0.])) / factor
                            self.assertLess(abs(one_wedge.kah - kah) / kah, rel_tol)

                            if one_wedge.kah - log_spiral.kah > 1e-5:  # in case the assumption of a logspiral is too
                                # restrictive, the one-wedge mechanism may be governing
                                # In that case, check at least that the logspiral mechanism can give the same solution
                                # as the one-wedge
                                log_spiral.optimize(
                                    np.array([half_pi - one_wedge.optimize_result.x[0] - log_spiral.params[PHI], 0]))
                                self.assertLess(abs(one_wedge.optimize_result.fun - log_spiral._external_energy(
                                    optimum_one_wedge)), 1e-5)
                            else:  # Generally, the logspiral mechanism should give a higher limit load than the one wedge
                                self.assertLess(one_wedge.kah - log_spiral.kah, 1e-5)
                        else:
                            print(
                                "The optimal configuration of the OneWedge mech. lies outside the admissible configurations for the LogSandwich. No test can be performed.")
                            print("K_ah{{1Wedge}} = {:.5f} K_ah{{LogSandwich}} = {:.5f}".format(one_wedge.kah,
                                                                                                log_spiral.kah))


if __name__ == '__main__':
    unittest.main()
