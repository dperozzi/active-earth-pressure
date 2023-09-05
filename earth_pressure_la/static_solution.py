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

import matplotlib.pyplot as plt
import numpy as np

from misc.definitions import Parameters, ALPHA, BETA, GAMMA, DELTA, PHI
from misc.exceptions import *
from misc.helpers import plot_point, plot_line, plot_circle, find_nearest


class StaticSolution:
    """
    A class for computing and visualizing static solutions for earth pressure problems.

    Attributes
    ----------
    active : bool
        Flag indicating if the solution is active or passive. Default is True.
    sign : float
        Sign used to calculate the position of the centre of the Mohr circle. Default is -1 if active=True else 1.
    params : dict
        Dictionary containing the parameters for the earth pressure problem.
    params_maxsize : int
        Size of the largest parameters array.
    h_soil : float
        Vertical height of the soil right behind the wall.
    k_n : numpy.ndarray
        Coefficient of earth pressure (in the normal direction) on the wall.
    theta_rankine : numpy.ndarray
        Direction of the principal stresses in the Rankine's stress region.
    theta_wall : numpy.ndarray
        Direction of the principal stresses in the region behind the wall.
    theta_logspiral : numpy.ndarray
        Angle of the logspiral shear zone.
    sigma_n : numpy.ndarray
        sigma_n in the Rankine's region.
    sigma_centre_rankine : numpy.ndarray
        Centre of the Mohr's circle in the Rankine's region.
    sigma_centre_wall : numpy.ndarray
        Centre of the Mohr's circle in the region behind the wall.

    Methods
    -------
    set_parameters(phi, delta, alpha, beta, gamma=1.)
        Sets the parameters for the earth pressure problem.
    set_parameter_by_name(name, value)
        Sets a specific parameter by name.
    compute()
        Computes the principal stresses and centres of the Mohr's circles.
    draw_mohr_circle(var_name=None, value=None, ax=None)
        Draws the Mohr's circle for a specific parameter.
    moment_wallbase
        Computes the moment at the base of the wall.
    earthpressure_n
        Computes the normal component of the earth pressure on the wall.
    earthpressure_res
        Computes the resultant earth pressure.
    earthpressure_h
        Computes the horizontal component of the earth pressure on the wall.
    k_res_norm
        Computes the normalized coefficient of earth pressure for the resultant force.
    k_h_norm
        Computes the normalized coefficient of earth pressure in horizontal direction.
    k_n_norm
        Computes the normalized coefficient of earth pressure in normal direction.
    """

    def __init__(self, active=True):
        self.active = active
        self.sign = -1. if self.active else 1.
        self.params = dict()
        self.params_maxsize = 0
        self.h_soil = 1.
        # Coefficient of earth pressure (in the normal direction) on the wall
        self.k_n = np.array([0.])
        # Direction of the principal stresses in the rankine's stress region
        self.theta_rankine = np.array([0.])
        # Direction of the principal stresses in the region behind the wall
        self.theta_wall = np.array([0.])
        # Angle of the logspiral shear zone
        self.theta_logspiral = np.array([0.])
        # sigma_n in the Rankine's region
        self.sigma_n = np.array([0.])
        # Centre of the Mohr's circle in the Rankine's region
        self.sigma_centre_rankine = np.array([0.])
        self.sigma_centre_wall = np.array([0.])

    def set_parameters(self, phi, delta, alpha, beta, gamma=1.):
        """
        Sets the parameters describing the geometry and the materials characterizing the BVP.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters.

        """
        self.params = {
            PHI: np.asarray(phi).reshape(-1),
            DELTA: np.asarray(delta).reshape(-1),
            ALPHA: np.asarray(alpha).reshape(-1),
            BETA: np.asarray(beta).reshape(-1),
            GAMMA: np.asarray(gamma).reshape(-1)
        }
        self._update_maxsize()
        self._expand_parameters()

    def set_parameter_by_name(self, name: Parameters, value):
        """
        Sets the parameter value by name.


        Parameters
        ----------
        par_name : Parameters
            The name of the parameter to be modified.
        value : Union[float, np.ndarray]
            The new value to be assigned to the parameter.

        """
        self.params[name] = value
        for param in Parameters:
            if param != name:
                self.params[param] = np.unique(self.params[param])
        self._update_maxsize()
        self._expand_parameters()

    def _update_maxsize(self):
        maxsize = 0
        for param in Parameters:
            this_size = self.params[param].size
            maxsize = max(this_size, maxsize)

        self.params_maxsize = maxsize

    def _expand_parameters(self):
        for param in Parameters:
            this_size = self.params[param].size
            if this_size < self.params_maxsize:
                if this_size == 1:
                    self.params[param] = np.repeat(self.params[param], self.params_maxsize)
                else:
                    raise InvalidConfiguration(
                        "The parameter {:s} has size {:d} which is less than the size of the parameter with the highest number of elements (n={:d}), but larger than one.".format(
                            param.name, this_size, self.params_maxsize))

    def compute(self):
        self.theta_rankine = 0.5 * (
                np.arcsin(np.divide(np.sin(self.params[BETA]), np.sin(self.params[PHI]))) - self.params[BETA])
        self.sigma_n = np.power(np.cos(self.params[BETA]), 2.)
        self.sigma_centre_rankine = np.divide(np.multiply(self.sigma_n, 1 + self.sign * np.sqrt(
            1 - np.multiply(np.power(np.cos(self.params[PHI]), 2), 1 + np.power(np.tan(self.params[BETA]), 2)))),
                                              np.power(np.cos(self.params[PHI]), 2.))

    def draw_mohr_circle(self, var_name=None, value=None, ax=None):
        """
        Draw the Mohr circle corresponding to the stress state for a given value of a chosen variable.

        Parameters
        ----------
        var_name : Parameters
            The name of the variable of interest.
        value : float
            The value of the variable for which the Mohr circle is to be drawn.
        ax : Axes object, optional
            The matplotlib Axes object on which the Mohr circle will be plotted.

        Returns
        -------
        ax : Axes object
            The matplotlib Axes object on which the Mohr circle was plotted.
        index_variable : int
            The index of the variable in the parameters array corresponding to the value provided.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        if var_name is not None:
            if value is None:
                raise ValueError("The variable `value` cannot be None")
            index_variable = find_nearest(self.params[var_name], value)[1]
        else:
            index_variable = 0

        phi = self.params[PHI][index_variable]
        delta = self.params[DELTA][index_variable]
        beta = self.params[BETA][index_variable]

        r_rankine = np.multiply(self.sigma_centre_rankine[index_variable], np.sin(phi))

        # Stress state in the infinite slope
        plot_point(ax, self.sigma_n[index_variable], self.sigma_n[index_variable] * np.tan(beta), "N", color='k')

        # Circle 1 (Rankine)
        plot_circle(ax, self.sigma_centre_rankine[index_variable], r_rankine, color='k')

        # Pole of circle 1
        sigma_p1 = self.sigma_centre_rankine[index_variable] - r_rankine * np.cos(
            2 * self.theta_rankine[index_variable])
        tau_p1 = r_rankine * np.sin(2 * self.theta_rankine[index_variable])
        plot_point(ax, sigma_p1, tau_p1, "P$_1$", offset=(5, -10), color='k')
        ax.plot([sigma_p1, self.sigma_centre_rankine[index_variable] + r_rankine], [tau_p1, 0], color='k', lw=0.9)

        # Failure line, backfill line, wall-friction line
        plot_line(ax, phi, color='k', label=r"$\varphi$")
        plot_line(ax, -phi, color='k')
        plot_line(ax, delta, color='k', ls='-.', label=r"$\delta$")
        plot_line(ax, -delta, color='k', ls='-.')
        plot_line(ax, beta, color='k', ls=':', label=r"$\beta$")
        plot_line(ax, -beta, color='k', ls=':')

        print("K_ah = {:.4f}".format(self.k_h_norm[index_variable]))
        print("K_an = {:.4f}".format(self.k_n_norm[index_variable]))
        print("K_a = {:.4f}".format(self.k_res_norm[index_variable]))
        print("E_ah = {:.4f}".format(self.earthpressure_h[index_variable]))
        print("E_a = {:.4f} (resultant force)".format(self.earthpressure_res[index_variable]))
        print("M_a = {:.4f} (resultant moment)".format(self.moment_wallbase[index_variable]))

        print("theta_1 = {:.1f}".format(np.rad2deg(self.theta_rankine[index_variable])))

        return ax, index_variable

    @property
    def moment_wallbase(self):
        """
        Calculate the moment exerted at the wall base.
        """
        alpha = self.params[ALPHA]
        delta = self.params[DELTA]
        const_factor = self.params[GAMMA] * self.h_soil ** 3. / 6.
        return np.multiply(const_factor, np.multiply(self.k_h_norm, np.divide(np.cos(delta),
                                                                              np.multiply(np.cos(alpha),
                                                                                          np.cos(alpha + delta)))))

    @property
    def earthpressure_n(self):
        """
        Calculate the normal component of the earth pressure exerted on the wall.
        """
        alpha = self.params[ALPHA]
        beta = self.params[BETA]
        const_factor = self.params[GAMMA] * self.h_soil ** 2. / 2.
        return np.multiply(const_factor, np.multiply(self.k_n, np.divide(np.cos(alpha - beta),
                                                                         np.multiply(np.power(np.cos(alpha), 2),
                                                                                     np.cos(beta)))))

    @property
    def earthpressure_res(self):
        """
        Calculate the resultant earth pressure exerted on the wall.
        """
        return np.divide(self.earthpressure_n, np.cos(self.params[DELTA]))

    @property
    def earthpressure_h(self):
        """
        Calculate the horizontal earth pressure exerted on the wall.
        """
        return np.multiply(self.k_h_norm, .5 * np.multiply(self.params[GAMMA], np.power(self.h_soil, 2)))

    @property
    def k_res_norm(self):
        """
        Normalized coefficient of earth pressure
        """
        return np.divide(self.k_n_norm, np.cos(self.params[DELTA]))

    @property
    def k_h_norm(self):
        """
        Normalized coefficient of earth pressure
        """
        factor = np.cos(self.params[ALPHA] + self.params[DELTA])
        factor = np.divide(factor, np.cos(self.params[DELTA]))
        return np.multiply(self.k_n_norm, factor)

    @property
    def k_n_norm(self):
        """
        Normalized coefficient of earth pressure
        """
        factor = np.cos(self.params[ALPHA] - self.params[BETA])
        factor = np.divide(factor, np.power(np.cos(self.params[ALPHA]), 2))
        factor = np.divide(factor, np.cos(self.params[BETA]))
        return np.multiply(self.k_n, factor)


class RankineActive(StaticSolution):
    """
    Rankine solution, formulated as a static solution of limit analysis
    """

    def __init__(self):
        super().__init__(True)

    def compute(self):
        super().compute()
        self.k_n = np.multiply(self.sigma_centre_rankine,
                               1 - np.multiply(np.sin(self.params[PHI]), np.cos(2 * self.theta_rankine)))

    def draw_mohr_circle(self, var_name=None, value=None, ax=None):
        _, _ = super().draw_mohr_circle(var_name, value, ax)


class LancellottaExtended(StaticSolution):
    """
    Extended Lancellotta's solution, as formulated in https://doi.org/10.3929/ethz-b-000591353 and mentioned in
    D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth pressure behind walls rotating about the base",
    submitted to Géotechnique in 2023.
    """

    def __init__(self):
        super().__init__(True)

    def compute(self):
        super().compute()
        self.theta_wall = 0.5 * (
                np.arcsin(np.divide(np.sin(self.params[DELTA]), np.sin(self.params[PHI]))) - self.params[DELTA]) - \
                          self.params[ALPHA]
        self.theta_logspiral = self.theta_wall - self.theta_rankine
        self.sigma_centre_wall = np.multiply(self.sigma_centre_rankine, np.exp(
            -2. * np.multiply(self.theta_logspiral, np.tan(self.params[PHI]))))
        two_chi = 2. * (self.theta_wall + self.params[ALPHA])
        self.k_n = np.multiply(self.sigma_centre_wall, 1. - np.multiply(np.sin(self.params[PHI]), np.cos(two_chi)))

    def draw_mohr_circle(self, var_name=None, value=None, ax=None):
        ax, index_variable = super().draw_mohr_circle(var_name, value, ax)

        phi = self.params[PHI][index_variable]
        delta = self.params[DELTA][index_variable]

        r_wall = np.multiply(self.sigma_centre_wall[index_variable], np.sin(phi))

        plot_circle(ax, self.sigma_centre_wall[index_variable], r_wall, color='.5')
        # Pole of circle 3 (Wall)
        sigma_p3 = self.sigma_centre_wall[index_variable] - r_wall * np.cos(2 * self.theta_wall[index_variable])
        tau_p3 = r_wall * np.sin(2. * self.theta_wall[index_variable])
        plot_point(ax, sigma_p3, tau_p3, "P$_3$", offset=(-5, 5), color='.5')
        ax.plot([sigma_p3, self.sigma_centre_wall[index_variable] + r_wall], [tau_p3, 0], color='.5', lw=0.9)

        sigma_xi = self.k_n[index_variable]
        tau_xi = -sigma_xi * np.tan(delta)
        plot_point(ax, sigma_xi, tau_xi, "$\\Xi$", offset=(-5, -10), color='.5')

        print("theta_3 = {:.1f}".format(np.rad2deg(self.theta_wall[index_variable])))
        print("theta = {:.1f}".format(np.rad2deg(self.theta_logspiral[index_variable])))
