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


from abc import ABC, abstractmethod

import numpy as np

from misc.exceptions import InvalidConfiguration


class Element(ABC):
    """
    Base class for defining elements in a kinematic failure mechanism for limit analysis.

    Attributes
    ----------
    vels : list
        List containing the absolute value of the velocity at the top and at the bottom of the element.
    thetas : list
        List containing angles describing the element and the direction of the velocity vector.
    lengths : list
        List containing the lenght of the element sides.

    Methods
    -------
    plot(ax, **plot_kwargs)
        Abstract method to plot the element.
    """

    def __init__(self):
        self.vels = []
        self.thetas = []
        self.lengths = []

    def _set_params_parent(self, v_bot, v_top):
        self.vels = [v_bot, v_top]

    @abstractmethod
    def plot(self, ax, **plot_kwargs):
        pass


class Wedge(Element):
    """
    Class representing a wedge element according to D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth
    pressure behind walls rotating about the base", submitted to Géotechnique in 2023. Inherits from the Element class.

    Methods
    -------
    set_params(self, t_1, t_2, l_1, t_v, v_bot, v_top)
        Set the parameters of the wedge.
    external_energy(self, gamma=1.)
        Calculate the external energy of the wedge.
    area(self)
        Calculate the area of the wedge.
    vertices(self)
        Get the vertices of the wedge.
    plot(self, ax, v_bot=np.zeros(2), ccw_rot=0., **plot_kwargs)
        Plot the wedge.
    """

    def __init__(self, t_1=0., t_2=np.pi / 2., l_1=1., t_v=0., v_bot=0., v_top=0.):
        """
        Class constructor.

        Parameters
        ----------
        t_1, t_2 : float, optional
            Angles representing the wedge geometry.
        l_1 : float, optional
            Length representing the wedge geometry.
        t_v : float, optional
            Inclination of the velocity vector to the horizontal.
        v_bot : float, optional
            Magnitude of the velocity at the bottom of the wedge.
        v_top : float, optional
            Magnitude of the velocity at the top of the wedge.
        """
        super().__init__()
        self.set_params(t_1, t_2, l_1, t_v, v_bot, v_top)

    def set_params(self, t_1, t_2, l_1, t_v, v_bot, v_top):
        """
        Set the parameters of the wedge according to the descriprion in the constructor and to Perozzi and Puzrin (2023)

        """
        super()._set_params_parent(v_bot, v_top)
        self.thetas = [t_1, t_2, t_v]
        self.lengths = [l_1, l_1 * np.sin(self.thetas[1]) / np.sin(self.thetas[0] + self.thetas[1])]

    def external_energy(self, gamma=1.):
        """
        Computes the external energy according to Perozzi and Puzrin (2023), Eq. (10)

        Parameters
        ----------
        gamma : float, optional
            The unit soil weight. Default is 1.

        Returns
        -------
        float
            The calculated external energy value.

        Raises
        ------
        InvalidConfiguration
            If division by zero or 0*infinity occurs during the calculation.
        """
        with np.errstate(all='raise'):
            try:
                w_e = gamma * np.sin(self.thetas[0]) * np.sin(self.thetas[1]) * np.sin(self.thetas[2]) / np.sin(
                    self.thetas[0] + self.thetas[1]) * self.lengths[0] ** 2 * (
                              1. / 3. * self.vels[0] + 1. / 6. * self.vels[1])
            except FloatingPointError as e:
                # Catch exception due to division by zero, or 0*infinity
                raise InvalidConfiguration(e)
        return w_e

    def area(self):
        """
        Compute the area of a wedge.
        """
        return 0.5 * self.lengths[0] * self.lengths[1] * np.sin(self.thetas[0])

    @property
    def vertices(self):
        l_1, l_2 = self.lengths
        t_1, t_2 = self.thetas[:2]
        return np.array([(0, 0), (l_2 * np.sin(t_1), l_2 * np.cos(t_1)), (0, l_1)])

    def plot(self, ax, v_bot=np.zeros(2), ccw_rot=0., **plot_kwargs):
        """
        Plot the wedge in a given axes object

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object on which to plot the wedge.
        v_bot : numpy.ndarray, optional
            The bottom vertex of the wedge. Defaults to a 2D array of zeros.
        ccw_rot : float, optional
            The counter-clockwise rotation angle in radians. Defaults to 0.
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the `plot` method of the Axes object.
        """
        R = np.array([[np.cos(ccw_rot), -np.sin(ccw_rot)], [np.sin(ccw_rot), np.cos(ccw_rot)]])
        points = np.dot(R, self.vertices.T) + v_bot.reshape(-1, 1)
        ax.plot(points[0, :], points[1, :], **plot_kwargs)


class LogSpiral(Element):
    """
    Class representing a logarithmic spiral sector element according to D. Perozzi, A. M. Puzrin,
    "Limit-state solutions for the active earth pressure behind walls rotating about the base", submitted to
    Géotechnique in 2023. Inherits from the Element class.

    Methods
    -------
    set_params(self, t_1, t_2, l_1, t_v, v_bot, v_top)
        Set the parameters of the wedge.
    external_energy(self, gamma=1.)
        Calculate the external energy of the wedge.
    area(self)
        Calculate the area of the wedge.
    vertices(self)
        Get the vertices of the wedge.
    plot(self, ax, v_bot=np.zeros(2), ccw_rot=0., **plot_kwargs)
        Plot the wedge.
    """
    def __init__(self, t_1=0., t_2=np.pi / 2., l_1=1., v_bot=0., v_top=0., phi=0.):
        super().__init__()
        self.phi = 0.
        self.set_params(t_1, t_2, l_1, v_bot, v_top, phi)

    def set_params(self, t_1, t_2, l_1, v_bot, v_top, phi):
        super()._set_params_parent(v_bot, v_top)
        self.thetas = [t_1, t_2]
        self.lengths = [l_1]
        self.phi = phi

    def external_energy(self, gamma=1.):
        return gamma * np.power(self.lengths[0], 2) * (self.vels[1] / 6. + self.vels[0] / 3.) / (
                1. + 9. * np.power(np.tan(self.phi), 2)) * (np.exp(3 * self.thetas[1] * np.tan(self.phi)) * (
                np.sin(self.thetas[0] + self.thetas[1]) + 3 * np.tan(self.phi) * np.cos(
            self.thetas[0] + self.thetas[1])) - np.sin(self.thetas[0]) - 3 * np.tan(self.phi) * np.cos(
            self.thetas[0]))

    def plot(self, ax, **plot_kwargs):
        t_1, t_2 = self.thetas
        chi = np.linspace(-t_1, -t_1 - t_2, 500)
        l_init = self.lengths[0]
        r = l_init * np.exp(-np.tan(self.phi) * (chi + t_1))
        ax.plot(np.multiply(r, np.cos(chi)), np.multiply(r, np.sin(chi)), **plot_kwargs)
        ax.plot([0, r[0] * np.cos(chi[0])], [0, r[0] * np.sin(chi[0])], **plot_kwargs)
