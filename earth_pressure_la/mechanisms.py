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

import math
from itertools import product

import numpy as np

from misc.definitions import ALPHA, BETA, GAMMA, DELTA, PHI
from misc.definitions import Bounds
from misc.definitions import LinearConstraint
from misc.exceptions import UnavailableFailureMode
from . import elements as el
from .base_mechanism import BaseMechanism

half_pi = np.pi * 0.5


class OneWedge(BaseMechanism):
    """
    A subclass of the BaseMechanism class, representing the single-wedge mechanism
    described in D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth pressure behind
    walls rotating about the base", submitted to Géotechnique in 2023.

    Attributes
    ----------
    elements : list
        A list of elements in the mechanism. This contains a single instance of
        the `Wedge` class.


    Methods
    -------
    optimize(self, x0=None)
        Optimizes the mechanism parameters using the specified optimization method.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.elements = [el.Wedge()]
        self.mode = mode.upper()
        self.optimize_method = "trust-constr"
        if self.mode != "T" and self.mode != "RF":
            raise UnavailableFailureMode(
                "Failure mode {:s} is not yet implemented (or is it just misspelled?).".format(self.mode))

    def optimize(self, x0=None):
        """
        Define bounds and initial guess (if it's not provided) and optimize the mechanism parameters calling the base
        class' method.

        Parameters
        ----------
        x0 : numpy.ndarray, optional
            Initial guess for the optimization.

        """
        if self.mode == "T":
            self.bounds = [
                Bounds([0.], [half_pi + self.params[ALPHA] - max(self.params[BETA], self.params[PHI])])]
        elif self.mode == "RF":
            self.bounds = [
                Bounds([0.], [min(half_pi + self.params[ALPHA] - self.params[BETA], np.pi - self.params[PHI])])]

        if self.optimize_method != "genetic":
            if x0 is None:
                self.x0 = np.array([(half_pi - self.params[PHI]) * 0.5 + self.params[ALPHA]])
            else:
                self.x0 = x0
            if self.x0[0] < self.bounds[0].lb[0] or self.x0[0] > self.bounds[0].ub[0]:
                self.x0[0] = .5 * (self.bounds[0].lb[0] + self.bounds[0].ub[0])
        self._optimize()

    def _update_mech_plot(self, x: np.ndarray):
        # Update the parameters describing the mechanism for plotting purposes
        super()._update_mech_plot(x)
        a = self.params[ALPHA]
        lw = self.h_soil / math.cos(a)
        self.element_config_plot = [(np.array((lw * math.sin(a), -lw * math.cos(a))), a)]

    def _update_mech(self, x: np.ndarray):
        # Update the mechanism-related parameters

        # Angles
        t_1 = x[0]
        t_2 = half_pi - self.params[ALPHA] + self.params[BETA]

        # Lengths
        l_1 = self.h_soil / np.cos(self.params[ALPHA])

        # Inclination and magnitude of the velocity vector
        if self.mode == "T":
            t_v = half_pi + self.params[ALPHA] - t_1 - self.params[PHI]
            v_t = np.cos(self.params[DELTA] + self.params[ALPHA]) / np.sin(
                self.params[DELTA] + t_1 + self.params[PHI])
            v_b = v_t
        elif self.mode == "RF":
            t_v = half_pi + self.params[ALPHA] - t_1 - self.params[PHI]
            v_b = 0
            if t_1 < half_pi - self.params[PHI]:
                delta = self.params[DELTA]
            else:
                delta = -self.params[DELTA]
            v_t = l_1 * np.cos(delta) / np.sin(delta + t_1 + self.params[PHI])
        else:
            t_v = v_t = v_b = 0.

        # Pass the correct configuration to the elements
        self.elements[0].set_params(t_1, t_2, l_1, t_v, v_b, v_t)


class TwoWedges(BaseMechanism):
    """
    A subclass of the BaseMechanism class, representing the two wedges mechanism
    described in D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth pressure behind
    walls rotating about the base", submitted to Géotechnique in 2023.

    Attributes
    ----------
    elements : list
        A list of elements in the mechanism. This contains two instances of
        the `Wedge` class.

    Methods
    -------
    optimize(self, x0=None)
        Optimizes the mechanism parameters using the specified optimization method.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.elements = [el.Wedge(), el.Wedge()]
        self.mode = mode.upper()
        if self.mode != "T" and self.mode != "RF":
            raise UnavailableFailureMode(
                "Failure mode {:s} is not yet implemented (or is it just misspelled?).".format(self.mode))

    def optimize(self, x0=None):
        """
        Define bounds, constraints, and initial guess (if it's not provided) and optimize the mechanism parameters 
        calling the base class' method.

        Parameters
        ----------
        x0 : numpy.ndarray, optional
            Initial guess for the optimization.

        """
        # These bounds and constraints are valid for all configurations
        bounds = Bounds([0., 0., 0.],
                        [half_pi + self.params[ALPHA], half_pi - self.params[ALPHA] + self.params[BETA],
                         np.pi])
        self.lconstr = LinearConstraint([0, 1, -1], [self.params[BETA] - self.params[ALPHA] - half_pi], [np.inf])

        if self.mode == "T":
            # Specialized bounds for wall translation
            bounds_spec = [
                [  # Wedge I
                    Bounds([max(self.params[ALPHA] - self.params[PHI] - half_pi,
                                -self.params[PHI] - self.params[DELTA]), -np.pi, -np.pi], [
                               min(half_pi + self.params[ALPHA] - self.params[PHI],
                                   np.pi - self.params[PHI] - self.params[DELTA]), np.pi, np.pi]),
                    Bounds([max(self.params[ALPHA] - self.params[PHI] + half_pi,
                                -self.params[PHI] + self.params[DELTA]), -np.pi, -np.pi], [
                               min(1.5 * half_pi + self.params[ALPHA] - self.params[PHI],
                                   np.pi - self.params[PHI] + self.params[DELTA]), np.pi, np.pi])],
                [  # Wedge II
                    Bounds([-np.inf, -np.inf, -2 * self.params[PHI]],
                           [np.inf, np.inf, np.pi - 2 * self.params[PHI]]),
                    Bounds([-np.inf, -np.inf, 0.], [np.inf, np.inf, np.pi])
                ]
            ]
        elif self.mode == "RF":
            bounds_spec = [
                [  # Wedge I
                    Bounds([max(- self.params[PHI] - half_pi, -self.params[PHI] - self.params[DELTA]), -np.pi,
                            -np.pi],
                           [min(half_pi - self.params[PHI], np.pi - self.params[PHI] - self.params[DELTA]), np.pi,
                            np.pi]),  # Concerning Fig. 6a
                    Bounds([max(- self.params[PHI] + half_pi, -self.params[PHI] + self.params[DELTA]), -np.pi,
                            -np.pi],  # Concerning Fig. 6b
                           [min(3. * half_pi - self.params[PHI], np.pi - self.params[PHI] + self.params[DELTA]),
                            np.pi, np.pi])],
                [  # Wedge II
                    Bounds([-np.inf, -np.inf, -2 * self.params[PHI]],
                           [np.inf, np.inf, np.pi - 2 * self.params[PHI]]),  # Concerning Fig. 6c
                    Bounds([-np.inf, -np.inf, 0.], [np.inf, np.inf, np.pi])  # Concerning Fig. 6d
                ]
            ]
        else:
            bounds_spec = []

        lconstr_spec = [
            [  # no specialized constraints for wedge I
                LinearConstraint([1, 1, 1], [-np.inf], [np.inf]),
                LinearConstraint([1, 1, 1], [-np.inf], [np.inf]),
            ],
            [  # specialized constraints for wedge II
                LinearConstraint([[1, 1, -1], [1, 1, 0]], [0., -2 * self.params[PHI]],
                                 [np.pi, np.pi - 2 * self.params[PHI]]),  # Concerning velocity diagram in Fig. 6c
                LinearConstraint([[1, 1, -1], [1, 1, 0]], [-np.pi, 0.], [0., np.pi]),  # Fig. 6d
            ]
        ]

        #  Generates the cartesian product of specialized bounds and constraints.
        #  The cartesian product of two lists is a list of all possible pairs you can form using elements
        #  of one and the other list.
        bounds_spec = list(product(*bounds_spec))
        self.lconstr_spec = list(product(*lconstr_spec))

        # Find the determinant bound (i.e. the most restrictive)
        self.bounds = []
        for bound_pair in bounds_spec:
            lb = bounds.lb
            ub = bounds.ub
            for bound in bound_pair:
                lb = np.maximum(lb, bound.lb)
                ub = np.minimum(ub, bound.ub)
            self.bounds.append(Bounds(lb, ub))

        # Set the initial guess if it is needed but not provided
        if self.optimize_method != "genetic":
            if x0 is None:
                theta = self.params[PHI] + np.arctan(np.cos(self.params[PHI] - self.params[ALPHA]) / (
                        np.sin(self.params[PHI] - self.params[ALPHA]) + np.sqrt(
                    np.sin(self.params[PHI] + self.params[DELTA]) * np.cos(
                        -self.params[BETA] + self.params[ALPHA]) / (
                            np.sin(self.params[PHI] - self.params[BETA]) * np.cos(
                        self.params[ALPHA] + self.params[DELTA])))))
                angle = half_pi + self.params[ALPHA] - theta
                t_2 = .5 * (half_pi + self.params[BETA] - self.params[ALPHA])
                self.x0 = np.array([angle, t_2, half_pi + self.params[ALPHA] + t_2 - angle])
            else:
                self.x0 = x0
        self._optimize()

    def _update_mech_plot(self, x: np.ndarray):
        # Update the parameters describing the mechanism for plotting purposes
        super()._update_mech_plot(x)
        a = self.params[ALPHA]
        lw = self.h_soil / math.cos(a)
        t_12 = x[1]
        self.element_config_plot = [(np.array((lw * math.sin(a), -lw * math.cos(a))), a),
                                    (np.array((self.elements[1].lengths[0] * math.sin(a + t_12),
                                               -self.elements[1].lengths[0] * math.cos(a + t_12))), a + t_12)]

    def _update_mech(self, x: np.ndarray):
        # Update the mechanism-related parameters
        # The velocities are denoted by v_i{b,t}, where i corresponds to the wedge number, b and t to the bottom or top
        # of the wedge. In case of a wall translation, the velocity at the bottom is the same as at the top.

        # Angles
        t_11 = x[0]
        t_12 = x[1]
        t_21 = x[2]
        t_22 = half_pi - self.params[ALPHA] - t_12 + self.params[BETA]

        # Lengths
        l_1 = self.h_soil / np.cos(self.params[ALPHA])
        l_2 = l_1 * np.sin(t_11) / np.sin(t_11 + t_12)

        # Inclination of the velocity vectors
        t_1v = half_pi + self.params[ALPHA] - t_11 - self.params[PHI]
        t_2v = half_pi + self.params[ALPHA] + t_12 - t_21 - self.params[PHI]

        # Magnitude of the velocity vectors
        if self.mode == "T":
            if t_11 < half_pi + self.params[ALPHA] - self.params[PHI]:
                delta = self.params[DELTA]
            else:
                delta = -self.params[DELTA]
            v_1t = np.cos(delta + self.params[ALPHA]) / np.sin(delta + t_11 + self.params[PHI])
            v_1b = v_1t
            if t_11 + t_12 - t_21 > 0:
                v_2t = v_1t * np.sin(t_11 + t_12 + 2 * self.params[PHI]) / np.sin(t_21 + 2 * self.params[PHI])
            else:
                v_2t = v_1t * np.sin(t_11 + t_12) / np.sin(t_21)
            v_2b = v_2t
        elif self.mode == "RF":
            if t_11 < half_pi - self.params[PHI]:
                delta = self.params[DELTA]
            else:
                delta = -self.params[DELTA]
            v_1t = l_1 * np.cos(delta) / np.sin(delta + t_11 + self.params[PHI])
            v_1b = 0
            if t_11 + t_12 - t_21 > 0:
                v_2t = v_1t * np.sin(t_11 + t_12 + 2 * self.params[PHI]) / np.sin(t_21 + 2 * self.params[PHI])
            else:
                v_2t = v_1t * np.sin(t_11 + t_12) / np.sin(t_21)
            v_2b = 0
        else:
            raise NotImplementedError("Failure modes other than 'T' and 'RF' have not been implemented yet.")

        # Pass the correct configuration to the elements
        self.elements[0].set_params(t_11, t_12, l_1, t_1v, v_1b, v_1t)
        self.elements[1].set_params(t_21, t_22, l_2, t_2v, v_2b, v_2t)


class ExtendedOneWedge(BaseMechanism):
    """
    A subclass of the BaseMechanism class, representing the extended one-wedge mechanism
    described in D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth pressure behind
    walls rotating about the base", submitted to Géotechnique in 2023.

    Attributes
    ----------
    one_wedge : OneWedge
        Wedge II in Fig. 3c in Perozzi and Puzrin (2023)
    elements : list
        A list of elements in the mechanism. This contains two instances of
        the `Wedge` class but is only used for plotting purposes in this class.
    vert_rigid_wedge : list
        A list containing the coordinates (in 2D) of the vertices of the wedge undergoing a rigid-body rotation
        (i.e. wedge I in Fig. 3c in Perozzi and Puzrin (2023))
    centroid_rigid_wedge : np.ndarray
        An array containing the coordinates of wedge I
    v_rigid_wedge : np.ndarray
        The velocity vector of wedge I (at the centroid)
    area_rigid_wedge : float
        The area of wedge I

    Methods
    -------
    optimize(self, x0=None)
        Optimizes the mechanism parameters using the specified optimization method.
    """

    def __init__(self, mode: str):
        # The actual wedge undergoing shearing
        self.one_wedge = OneWedge(mode)
        super().__init__()
        self.mode = mode
        if self.mode != "T" and self.mode != "RF":
            raise UnavailableFailureMode(
                "Failure mode {:s} is not yet implemented (or is it just misspelled?).".format(self.mode))
        self.elements = [el.Wedge(), el.Wedge()]  # Only used to plot the mechanism
        # The rigid wedge undergoing rigid body rotation together with the wall
        self.vert_rigid_wedge = [np.zeros((2,)), np.zeros((2,))]
        self.centroid_rigid_wedge = np.array([])
        self.v_rigid_wedge = np.array([])
        self.area_rigid_wedge = 0.

    def set_parameters(self, phi, delta, alpha, beta, gamma=None):
        super().set_parameters(phi, delta, alpha, beta, gamma)
        self.one_wedge.set_parameters(phi, delta, alpha, beta, gamma)

    def set_parameter_by_name(self, name: str, value):
        super().set_parameter_by_name(name, value)
        self.one_wedge.set_parameter_by_name(name, value)

    def optimize(self, x0=None):
        """
        Define bounds, constraints, and initial guess (if it's not provided) and optimize the mechanism parameters
        calling the base class' method.

        Parameters
        ----------
        x0 : numpy.ndarray, optional
            Initial guess for the optimization.

        """
        bounds = Bounds([0., 0.],
                        [half_pi + self.params[ALPHA] - self.params[BETA],
                         half_pi + self.params[ALPHA] - self.params[BETA]])
        self.lconstr = LinearConstraint(np.array([[1, 1]]), [0],
                                        [half_pi + self.params[ALPHA] - self.params[
                                            PHI]])  # PHI is more restrictive than BETA

        self.bounds = [bounds]

        if self.optimize_method != "genetic":
            if x0 is None:
                self.one_wedge.optimize()
                self.x0 = np.array([0., self.one_wedge.optimize_result.x[0]])
            else:
                self.x0 = x0

        self._optimize()

    def _update_mech(self, x: np.ndarray):
        # Update the mechanism-related parameters

        # Angle
        t_11 = x[0]

        # Lengths
        l_1 = self.h_soil / np.cos(self.params[ALPHA])
        l_2 = l_1 * np.cos(self.params[ALPHA] - self.params[BETA]) / np.cos(
            self.params[ALPHA] - self.params[BETA] - t_11)

        alpha_one_wedge = self.params[ALPHA] - t_11

        ab_length = self.h_soil * np.sin(t_11) / (
                np.cos(self.params[ALPHA]) * np.cos(t_11 - self.params[ALPHA] + self.params[BETA]))

        self.vert_rigid_wedge[0] = np.array([-self.h_soil * np.tan(self.params[ALPHA]), self.h_soil])
        self.vert_rigid_wedge[1] = self.vert_rigid_wedge[0] + ab_length * np.array(
            [np.cos(self.params[BETA]), np.sin(self.params[BETA])])
        self.centroid_rigid_wedge = np.array([1. / 3. * (self.vert_rigid_wedge[0][0] + self.vert_rigid_wedge[1][0]),
                                              1. / 3. * (self.vert_rigid_wedge[0][1] + self.vert_rigid_wedge[1][1])])
        if self.mode == "RF":
            self.v_rigid_wedge = -np.cross(self.centroid_rigid_wedge, np.array([0, 0, 1]))
        else:
            self.v_rigid_wedge = np.array([-1., 0.])
        self.area_rigid_wedge = 0.5 * np.abs(np.linalg.det(np.vstack(self.vert_rigid_wedge).T))

        self.one_wedge.set_parameter_by_name(ALPHA, alpha_one_wedge)
        if abs(t_11) < 1e-6:
            self.one_wedge.set_parameter_by_name(DELTA, self.params[DELTA])
        else:
            self.one_wedge.set_parameter_by_name(DELTA, self.params[PHI])
        self.one_wedge.h_soil = l_2 * np.cos(alpha_one_wedge)

    def _external_energy(self, x: np.ndarray):
        # Calculate the external energy for this mechanism, considering its custom attributes storing the wedges
        if type(x) is not np.ndarray:
            x = np.array(x)
        self._update_mech(x)
        energy = self.area_rigid_wedge * self.params[GAMMA] * self.v_rigid_wedge[1]
        energy += self.one_wedge._external_energy(x[1:])

        return energy

    def _update_mech_plot(self, x: np.ndarray):
        # Update the parameters describing the mechanism for plotting purposes
        super()._update_mech_plot(x)
        a = self.params[ALPHA]
        b = self.params[BETA]
        lw = self.h_soil / math.cos(a)
        vert = np.array((lw * math.sin(a), -lw * math.cos(a)))
        self.element_config_plot = [(vert, a), (vert, a - x[0])]
        self.elements[0].set_params(x[0], half_pi - a + b, lw, 0, 0, 0)
        self.elements[1].set_params(x[1], half_pi + x[0] - a + b, self.one_wedge.h_soil / np.cos(a - x[0]), 0, 0, 0)


class LogSpiral(BaseMechanism):
    """
    A subclass of the BaseMechanism class, representing the wedge-logarithmic spiral-wedge mechanism
    described in D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth pressure behind
    walls rotating about the base", submitted to Géotechnique in 2023.

    Attributes
    ----------
    elements : list
        A list of elements in the mechanism. This contains two instances of
        the `Wedge` class separated by one instance of the `LogSpiral` class.

    Methods
    -------
    optimize(self, x0=None)
        Optimizes the mechanism parameters using the specified optimization method.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.elements = [el.Wedge(), el.LogSpiral(), el.Wedge()]
        self.mode = mode.upper()
        if self.mode != "T" and self.mode != "RF":
            raise UnavailableFailureMode(
                "Failure mode {:s} is not yet implemented (or is it just misspelled?).".format(self.mode))
        self.optimize_method = "genetic"

    def optimize(self, x0=None):
        """
        Define bounds, constraints, and initial guess (if it's not provided) and optimize the mechanism parameters
        calling the base class' method.

        Parameters
        ----------
        x0 : numpy.ndarray, optional
            Initial guess for the optimization.

        """
        self.lconstr = LinearConstraint([1, 1], [0.], [half_pi - self.params[ALPHA] + self.params[BETA]])

        if self.mode == "T":
            self.bounds = [Bounds([max(0., -self.params[ALPHA]), 0.],
                                  [min(half_pi - self.params[ALPHA] + self.params[BETA],
                                       half_pi + self.params[DELTA]),
                                   half_pi - self.params[ALPHA] + self.params[BETA]])]
        elif self.mode == "RF":
            self.bounds = [Bounds([0., 0.],
                                  [min(half_pi - self.params[ALPHA] + self.params[BETA],
                                       half_pi + self.params[DELTA]),
                                   half_pi - self.params[ALPHA] + self.params[BETA]])]
        else:
            self.bounds = []

        if self.optimize_method != "genetic":
            if x0 is None:
                zeta_1 = self.params[PHI] + np.arctan(np.cos(self.params[PHI] - self.params[ALPHA]) / (
                        np.sin(self.params[PHI] - self.params[ALPHA]) + np.sqrt(
                    np.sin(self.params[PHI] + self.params[DELTA]) * np.cos(
                        -self.params[BETA] + self.params[ALPHA]) / (
                            np.sin(self.params[PHI] - self.params[BETA]) * np.cos(
                        self.params[ALPHA] + self.params[DELTA])))))
                t_2 = zeta_1 - self.params[ALPHA] - self.params[PHI]
                self.x0 = np.array([t_2, np.deg2rad(.5)])
            else:
                self.x0 = x0
        self._optimize()

    def _update_mech_plot(self, x: np.ndarray):
        # Update the parameters describing the mechanism for plotting purposes
        super()._update_mech_plot(x)
        a = self.params[ALPHA]
        lw = self.h_soil / math.cos(a)
        t_2nd_wedge = a + x[0] + x[1]
        self.element_config_plot = [(np.array((lw * math.sin(a), -lw * math.cos(a))), a), None,
                                    (np.array((self.elements[2].lengths[0] * math.sin(t_2nd_wedge),
                                               -self.elements[2].lengths[0] * math.cos(t_2nd_wedge))), t_2nd_wedge)]

    def _update_mech(self, x: np.ndarray):
        # Update the mechanism-related parameters

        # Angles
        t_11 = half_pi - x[0] - self.params[PHI]
        t_12 = x[0]
        t_21 = half_pi - self.params[ALPHA] - x[0] - x[1]
        t_22 = x[1]
        t_31 = half_pi - self.params[PHI]
        t_32 = half_pi - self.params[ALPHA] - x[0] - x[1] + self.params[BETA]

        # Lengths
        l_1 = self.h_soil / np.cos(self.params[ALPHA])
        l_12 = l_1 * np.sin(t_11) / np.sin(t_11 + t_12)
        l_23 = l_12 * np.exp(-x[1] * np.tan(self.params[PHI]))

        # Inclination of the velocity vectors
        t_1v = self.params[ALPHA] + x[0]
        t_3v = self.params[ALPHA] + x[0] + x[1]
        if self.mode == "T":
            v_1t = np.cos(self.params[ALPHA] + self.params[DELTA]) / np.cos(self.params[DELTA] - x[0])
            v_1b = v_1t
            v_3t = v_1t * np.exp(-x[1] * np.tan(self.params[PHI]))
            v_3b = v_3t
        elif self.mode == "RF":
            v_1t = l_1 * np.cos(self.params[DELTA]) / np.cos(self.params[DELTA] - x[0])
            v_1b = 0
            v_3t = v_1t * np.exp(-x[1] * np.tan(self.params[PHI]))
            v_3b = 0
        else:
            t_1v = v_1t = v_1b = 0.
            t_3v = v_3t = v_3b = 0.

        # Pass the correct configuration to the elements
        self.elements[0].set_params(t_11, t_12, l_1, t_1v, v_1b, v_1t)
        self.elements[1].set_params(t_21, t_22, l_23, v_3b, v_3t, self.params[PHI])
        self.elements[2].set_params(t_31, t_32, l_23, t_3v, v_3b, v_3t)
