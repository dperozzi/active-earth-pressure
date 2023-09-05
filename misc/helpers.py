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

import builtins

import numpy as np


def ka_coulomb(phi, delta, alpha, beta):
    """
    Coulomb's coefficient of active earth pressure.

    Parameters
    ----------
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    alpha = -alpha
    return np.cos(phi + alpha) ** 2 / (np.cos(alpha) ** 2 * np.cos(delta - alpha) * (1 + np.sqrt(
        np.sin(phi + delta) * np.sin(phi - beta) / (np.cos(delta - alpha) * np.cos(alpha + beta)))) ** 2)


def kah_coulomb(phi, delta, alpha, beta):
    """
    Coulomb's coefficient of active earth pressure (horizontal component).

    Parameters
    ----------
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    return ka_coulomb(phi, delta, alpha, beta) * np.cos(delta + alpha)


def kan_coulomb(phi, delta, alpha, beta):
    """
    Coulomb's coefficient of active earth pressure (normal component).

    Parameters
    ----------
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    return ka_coulomb(phi, delta, alpha, beta) * np.cos(delta)


def ea_coulomb(gamma, h, phi, delta, alpha, beta):
    """
    Coulomb's resultant active earth pressure.

    Parameters
    ----------
    gamma : float
        Soil unit weight.
    h : float
        Wall height.
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    return 0.5 * h ** 2 * gamma * ka_coulomb(phi, delta, alpha, beta)


def eah_coulomb(gamma, h, phi, delta, alpha, beta):
    """
    Coulomb's horizontal component of the active earth pressure.

    Parameters
    ----------
    gamma : float
        Soil unit weight.
    h : float
        Wall height.
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    return 0.5 * h ** 2 * gamma * kah_coulomb(phi, delta, alpha, beta)


def ma_coulomb(gamma, h, phi, delta, alpha, beta):
    """
    Moment exerted on the wall assuming Coulomb's active earth pressure.

    Parameters
    ----------
    gamma : float
        Soil unit weight.
    h : float
        Wall height.
    phi : float
        Soil friction angle.
    delta : float
        Interface friction angle.
    alpha : float
        Wall inclination.
    beta : float
        Backfill inclination.

    """
    return kan_coulomb(phi, delta, alpha, beta) * gamma * h ** 3 / (6. * np.cos(alpha))


def coulomb_la_manual(ph, d, a, b, t_1):
    """
    The active earth pressure based on the kinematic method of limit analysis under Coulomb's assumptions
    (straight failure line, only one wedge). The earth pressure is a function of the inclination of the failure
    line theta.

    Parameters
    ----------
    ph : float
        Soil friction angle phi.
    d : float
        Interface friction angle delta.
    a : float
        Wall inclination alpha.
    b : float
        Backfill inclination beta.
    t_1 : float
        Inclination of the failure mechanism theta.

    Returns
    -------
    float
        The result of the Coulomb LA calculation.
    """
    return np.cos(a - b) * np.sin(t_1) * np.cos(d + a) * np.cos(a - t_1 - ph) / (
            np.cos(a) ** 2 * np.cos(t_1 + b - a) * np.sin(d + t_1 + ph))


def plot_circle(ax, c_x, r, **mpl_kwargs):
    """
    Plot a circle in the given axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the circle will be plotted.

    c_x : float
        The x-coordinate of the center point of the circle.

    r : float
        The radius of the circle.

    **mpl_kwargs : dict
        Additional keyword arguments to be passed to the `plot` function.

    """
    t = np.linspace(0, 2 * np.pi, 150)
    ax.plot(r * np.cos(t) + c_x, r * np.sin(t), lw=1.1, **mpl_kwargs)


def plot_line(ax, angle, **mpl_kwargs):
    """
    Plot a line in a given axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the line.

    angle : float
        The angle in radians at which the line should be drawn.

    **mpl_kwargs : dict
        Additional keyword arguments to pass to the `plot` function.

    """
    ax.plot([0, ax.get_xlim()[1]], [0, np.tan(angle) * ax.get_xlim()[1]], lw=.9, **mpl_kwargs)


def plot_point(ax, x, y, label, offset=(5, 5), **mpl_kwargs):
    """
    Plot a point in a given axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the point will be plotted.
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    label : str
        The label associated with the point.
    offset : tuple, optional
        The offset in points to position the label relative to the point. Default is (5, 5).
    **mpl_kwargs : dict
        Additional keyword arguments to pass to the `scatter` function.

    """
    ax.scatter(x, y, s=10, zorder=2.5, **mpl_kwargs)
    ax.annotate(label,
                (x, y),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=offset,  # distance from text to points (x,y)
                ha='center',
                bbox=dict(boxstyle='square,pad=0.', facecolor="white", edgecolor="none"))


def find_nearest(array, value):
    """
    Parameters
    ----------
    array : ndarray
        The input array.

    value : float
        The value to find the nearest element to.

    Returns
    -------
    nearest_value : float
        The nearest value to the given value in the array.

    index : int
        The index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def run_from_ipython():
    """
    Check if the Python code is being run from IPython.

    Returns:
        bool: True if running from IPython, False otherwise.
    """
    if hasattr(builtins, '__IPYTHON__'):
        return True
    else:
        return False


def run_in_ipython_nb():
    """
    Checks if the code is running in an IPython notebook.

    Returns:
        bool: True if running in an IPython notebook, False otherwise.
    """
    if run_from_ipython():
        from IPython import get_ipython
        ip = get_ipython()
        if ip.has_trait('kernel'):
            return True
    return False
