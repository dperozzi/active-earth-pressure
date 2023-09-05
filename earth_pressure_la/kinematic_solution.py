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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misc.definitions import ONE_WEDGE, TWO_WEDGES, EXT_ONE_WEDGE, LOG_SANDWICH
from misc.definitions import Parameters, ALPHA, BETA, GAMMA, DELTA, PHI
from misc.helpers import kah_coulomb, kan_coulomb, find_nearest
from . import mechanisms as mec


class KinematicSolution:
    """
    The KinematicSolution provides a kinematic solution for the active earth pressure problem considering different
    possible failure mechanisms as described in D. Perozzi, A. M. Puzrin, "Limit-state solutions for the active earth
    pressure behind walls rotating about the base", submitted to Géotechnique in 2023.

    It allows for optimizing the mechanisms based on specified parameters and plotting the optimized configurations.

    Parameters
    ----------
    mechanisms : list
        List of mechanism types to be considered.
    parameters : dict
        Dictionary of parameters for the mechanisms.
    mode : str
        String specifying the mode of analysis.

    Attributes
    ----------
    mechanisms : dict
        Dictionary containing the mechanism types to be considered.
    params : dict
        Dictionary containing the parameters describing the geometry and the material properties of the
         boundary value problem.
    var_parameter : Parameters
        Denotes the variable parameter considered in the current analysis. One could seek the solution for a
        given (constant) parameter set, but considering one variable parameter.
        For example, phi=30, delta=20, alpha=0, beta=var.
    sec_var_parameters : list or None
        Denotes possible secondary variable parameters for the analysis. For example, one could consider a variable
        soil friction angle phi, but wanting to keep delta at a constant ratio (e.g. delta=2/3*phi) instead of
         a constant value.
    optimized_config : dict
        Dictionary containing the optimized configurations for each mechanism type.
    ka : dict
        Dictionary containing the coefficient of earth pressure K_a for each mechanism type and each parameter value.
    kan : dict
        Dictionary containing the coefficient of earth pressure K_an (relating to the normal component of the
        earth pressure on the wall) for each mechanism type and each parameter value.
    kah : dict
        Dictionary containing the coefficient of earth pressure K_ah (relating to the horizontal component of the
        earth pressure on the wall) for each mechanism type and each parameter value.
    kan_coulomb : dict
        Dictionary containing the coefficient of earth pressure K_an (relating to the normal component of the
        earth pressure on the wall) according to Coulomb's theory for each mechanism type and each parameter value.
    kah_coulomb : dict
        Dictionary containing the coefficient of earth pressure K_ah (relating to the horizontal component of the
        earth pressure on the wall) according to Coulomb's theory for each mechanism type and each parameter value.
    crit_mech : dict
        Dictionary containing the critical mechanism type for each parameter value.
    mech_names : dict
        Dictionary mapping mechanism types to mechanism names.
    var_labels : dict
        Dictionary mapping parameter names to parameter labels for plotting.

    Methods
    -------
    set_var_parameter(var_parameter: Parameters):
        Sets the variable parameter for optimization.
    set_secondary_var_parameters(var_parameters: list):
        Sets the secondary variable parameters for optimization.
    set_parameters(parameters: dict):
        Sets the parameters describing the geometry and the materials characterizing the BVP.
    set_parameter_by_name(par_name: Parameters, value):
        Sets the parameter value by name.
    optimize(ax=None, plot_values=None, plot_kwargs=None, figsize=None):
        Optimizes the mechanisms based on the variable parameter and plots the
        optimized configurations.

    Returns
    -------
    fig : matplotlib Figure, optional, default: None
        The first of the optional tuple returned by the `optimize` function.
    ax : matplotlib Axis, optional, default: None
        The second of the optional tuple returned by the `optimize` function.

    Examples
    --------
    >>> import numpy as np
    >>> from earth_pressure_la.kinematic_solution import KinematicSolution
    >>> beta = np.deg2rad(np.linspace(-30,30,61))
    >>> phi = np.radians(30)
    >>> delta = np.radians(20)
    >>> alpha = 0.
    >>> params = {PHI: phi, ALPHA:alpha, BETA:beta, DELTA:delta}
    >>> kin_sol = KinematicSolution([ONE_WEDGE, TWO_WEDGES, EXT_ONE_WEDGE], params, "RF")
    >>> kin_sol.set_var_parameter(BETA)
    >>> kin_sol.optimize()

    """

    def __init__(self, mechanisms: list, parameters: dict, mode: str):
        self.mechanisms = {}
        for mech in mechanisms:
            if mech == ONE_WEDGE:
                self.mechanisms[ONE_WEDGE] = mec.OneWedge(mode)
            elif mech == TWO_WEDGES:
                self.mechanisms[TWO_WEDGES] = mec.TwoWedges(mode)
            elif mech == EXT_ONE_WEDGE:
                self.mechanisms[EXT_ONE_WEDGE] = mec.ExtendedOneWedge(mode)
            elif mech == LOG_SANDWICH:
                self.mechanisms[LOG_SANDWICH] = mec.LogSpiral(mode)

        self.params = parameters
        self.var_parameter = None
        self.sec_var_parameters = None
        self.optimized_config = {}
        self.ka = {}
        self.kan = {}
        self.kah = {}
        self.kan_coulomb = {}
        self.kah_coulomb = {}
        self.crit_mech = {}
        self.mech_names = {
            ONE_WEDGE: "One wedge",
            TWO_WEDGES: "Two wedges",
            EXT_ONE_WEDGE: "Ext. one wedge",
            LOG_SANDWICH: "Logsandwich"
        }
        self.var_labels = {
            ALPHA: "$\\alpha={:.0f}^\circ$",
            BETA: "$\\beta={:.0f}^\circ$",
            DELTA: "$\\delta={:.0f}^\circ$",
            GAMMA: "$\\gamma={:.0f}\,\\mathrm{kN/m^3}$",
            PHI: "$\\varphi={:.0f}^\circ$"
        }

    def set_var_parameter(self, var_parameter: Parameters):
        """
        Sets the variable parameter for optimization.

        Parameters
        ----------
        var_parameter: Parameters
            The variable parameter to be set as variable in the KinematicSolution object.

        """
        self.var_parameter = var_parameter
        self.params[self.var_parameter] = np.atleast_1d(self.params[self.var_parameter])

    def set_secondary_var_parameters(self, var_parameters: list):
        """
        Sets the secondary variable parameters for optimization.

        Parameters
        ----------
        var_parameters : list
            A list of secondary variable parameters.

        """
        self.sec_var_parameters = var_parameters
        for sec_var_par in self.sec_var_parameters:
            self.params[sec_var_par] = np.array(self.params[sec_var_par])

    def set_parameters(self, parameters: dict):
        """
        Sets the parameters describing the geometry and the materials characterizing the BVP.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters.

        """
        self.params = parameters

    def set_parameter_by_name(self, par_name: Parameters, value):
        """
        Sets the parameter value by name.


        Parameters
        ----------
        par_name : Parameters
            The name of the parameter to be modified.
        value : Union[float, np.ndarray]
            The new value to be assigned to the parameter.

        """
        self.params[par_name] = value

    def optimize(self, ax=None, plot_values=None, plot_kwargs=None, figsize=None):
        """
        Optimizes the kinematic solution for a given set of parameters.

        Parameters
        ----------
        ax : AxesSubplot, optional
            The subplot to plot the optimized mechanism on. If `plot_values` is provided but `ax` is not, a new subplot will be created.
        plot_values : list of float, optional
            The values at which to plot the optimized mechanism. If None, no plotting will be done.
        plot_kwargs : list of dict, optional
            The keyword arguments for customizing the plot of each value in `plot_values`.
        figsize : tuple of float, optional
            The size of the figure for the subplot.

        Returns
        -------
        fig : Figure or None
            The figure containing the subplot if `plot_values` is provided, otherwise None.
        ax : AxesSubplot or None
            The subplot where the optimized mechanism is plotted if `plot_values` is provided, otherwise None.
        """
        if plot_values is not None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        # Automatically create a label for the plot as `variable_name=value` if no label was given
        if plot_values is not None:
            plot_values.sort()
            for i, val in enumerate(plot_values):
                if "label" not in plot_kwargs[i].keys():
                    if self.var_parameter is not GAMMA:
                        plot_kwargs[i]["label"] = self.var_labels[self.var_parameter].format(math.degrees(val))
                    else:
                        plot_kwargs[i]["label"] = self.var_labels[self.var_parameter].format(val)
            id_vals_plot = [find_nearest(self.params[self.var_parameter], val)[1] for val in plot_values]
        else:
            id_vals_plot = []

        ka_df = pd.DataFrame()
        kan_df = pd.DataFrame()
        kah_df = pd.DataFrame()
        optim_config_df = pd.DataFrame()

        for mech in self.mechanisms.values():
            for name, val in self.params.items():
                if name != self.var_parameter:
                    mech.set_parameter_by_name(name, val)

        # Optimize mechanism for each variable parameter
        for i, val in enumerate(self.params[self.var_parameter]):
            ka = {}
            kan = {}
            kah = {}
            optimized_config = {}
            for mech_type, mech in self.mechanisms.items():
                mech.set_parameter_by_name(self.var_parameter, val)
                if self.sec_var_parameters:  # if secondary variable parameters are present, change them as defined
                    for var in self.sec_var_parameters:
                        mech.set_parameter_by_name(var, self.params[var][i])
                if i == 0:
                    mech.optimize()
                else:
                    mech.optimize(mech.optimize_result.x)

                # Book keeping
                ka[mech_type] = mech.ka
                kan[mech_type] = mech.kan
                kah[mech_type] = mech.kah
                optimized_config[mech_type] = mech.optimize_result.x

            ka_df = ka_df.append(ka, ignore_index=True)
            kan_df = kan_df.append(kan, ignore_index=True)
            kah_df = kah_df.append(kah, ignore_index=True)
            optim_config_df = optim_config_df.append(optimized_config, ignore_index=True)

            if id_vals_plot and i == id_vals_plot[0]:
                crit_mech = kan_df.iloc[-1, :].round(7).idxmax()
                mech = self.mechanisms[crit_mech]
                if plot_kwargs is not None:
                    mech.plot_optimized_mech(ax, **plot_kwargs[0])
                    plot_kwargs = plot_kwargs[1:]
                else:
                    mech.plot_optimized_mech(ax)
                id_vals_plot = id_vals_plot[1:]

        # Reindex dataframes containing results
        ka_df.index = self.params[self.var_parameter]
        kan_df.index = self.params[self.var_parameter]
        kah_df.index = self.params[self.var_parameter]

        # Coefficient of earth pressure (resultant force)
        self.ka[self.var_parameter] = ka_df
        # Coefficient of earth pressure (normal component)
        self.kan[self.var_parameter] = kan_df
        # Coefficient of earth pressure (horizontal component)
        self.kah[self.var_parameter] = kah_df
        # Save the optimized configuration
        self.optimized_config[self.var_parameter] = optim_config_df

        # Check which one of the mechanisms is the most critical (i.e. leading to the highest load)
        self.crit_mech[self.var_parameter] = kan_df.round(7).idxmax(axis=1)
        self.crit_mech[self.var_parameter] = self.crit_mech[self.var_parameter].map(self.mech_names)

        # Calculate the same values according to Coulomb's Theory (single wedge, translation)
        self.kah_coulomb[self.var_parameter] = kah_coulomb(self.params[PHI], self.params[DELTA],
                                                           self.params[ALPHA], self.params[BETA])
        self.kan_coulomb[self.var_parameter] = kan_coulomb(self.params[PHI], self.params[DELTA],
                                                           self.params[ALPHA], self.params[BETA])

        return fig, ax
