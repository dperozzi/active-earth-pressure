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

import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interact

from . import static_solution as sts


class InteractiveStaticSol:
    """
    This class represents an interactive tool for drawing Mohr circles from a static solution.

    Attributes
    ----------
    fig : object
        The figure object used for plotting the Mohr circles.
    ax : object
        The axes object used for setting the plot properties.
    alpha_slider : object
        A float slider widget for adjusting the alpha parameter.
    beta_slider : object
        A float slider widget for adjusting the beta parameter.
    delta_slider : object
        A float slider widget for adjusting the delta parameter.
    phi_slider : object
        A float slider widget for adjusting the phi parameter.
    static_solution : LancellottaExtended Instance
        An instance of the LancellottaExtended class for performing the static solution computations.

    Methods
    -------
    update_delta_range()
        Updates the maximum value of the delta slider based on the current value of the phi slider.
    update_beta_range()
        Updates the minimum and maximum values of the beta slider based on the current value of the phi slider.
    draw_mohr_circles()
        Draws the corresponding Mohr circles based on the selected values of the sliders.
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.alpha_slider = widgets.FloatSlider(min=-45., max=45., step=1., value=0.)
        self.beta_slider = widgets.FloatSlider(min=-30., max=30., step=1., value=10.)
        self.delta_slider = widgets.FloatSlider(min=0., max=55., step=1., value=20.)
        self.phi_slider = widgets.FloatSlider(min=1., max=55., step=1., value=30.)

        self.phi_slider.observe(self.update_delta_range, 'value')
        self.phi_slider.observe(self.update_beta_range, 'value')

        self.static_solution = sts.LancellottaExtended()
        self.static_solution.set_parameters(math.radians(30), math.radians(20), math.radians(0), math.radians(20))

        interact(self.draw_mohr_circles, alpha=self.alpha_slider, beta=self.beta_slider, delta=self.delta_slider,
                 phi=self.phi_slider)

    def update_delta_range(self, *args):
        self.delta_slider.max = self.phi_slider.value

    def update_beta_range(self, *args):
        self.beta_slider.min = -self.phi_slider.value
        self.beta_slider.max = self.phi_slider.value

    def draw_mohr_circles(self, alpha, beta, delta, phi):
        alpha = math.radians(alpha)
        beta = math.radians(beta)
        delta = math.radians(delta)
        phi = math.radians(phi)
        self.ax.clear()
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([-.8, .8])
        self.ax.grid(True, linewidth=0.5)
        self.ax.set_aspect(1)
        self.ax.set_xlabel("$\\sigma/(\\gamma\\cdot z)$ [-]")
        self.ax.set_ylabel("$\\tau/(\\gamma\\cdot z)$ [-]")

        self.static_solution.set_parameters(phi, delta, alpha, beta)
        self.static_solution.compute()
        self.static_solution.draw_mohr_circle(ax=self.ax)
        self.ax.legend()
        self.fig.tight_layout()
