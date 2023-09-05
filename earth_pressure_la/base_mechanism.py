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

import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Union, Iterable

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

from misc.definitions import Bounds, LinearConstraint
from misc.definitions import Parameters, ALPHA, BETA, GAMMA, DELTA, PHI
from misc.exceptions import InvalidConfiguration, InconsistentConstraints, UnavailableFailureMode


class BaseMechanism(ABC):
    """
    Base class for a generic mechanism.

    Attributes
    ----------
    modes : tuple
        Tuple containing available failure modes.
    optimize_result : OptimizeResult, optional
        Result of the optimization.
    optimized_config : array-like
        The optimized mechanism configuration.
    optim_options : dict
        Dictionary of options for each optimization method.
    resultant_force : float
        The resultant force of the optimized mechanism configuration.
    ka : float
        The value of ka for the mechanism.
    kah : float
        The value of kah for the mechanism.
    kan : float
        The value of kan for the mechanism.
    elements : list
        List storing the elementary components that make up the mechanism, such as wedges and logarithmic spiral sectors.
    element_config_plot : list
        List of configurations for plotting each element (only needed for wedge elements.
    bounds : list
        List of bounds for the optimization variables.
    lconstr : LinearConstraint, optional
        Linear constraint for the optimization variables valid for each possible configuration of the mechanism.
    lconstr_spec : list, optional
        List of specialized linear constraints for the optimization variables. This list contains one list of
        specialized constraints per each element of the mechanism. The specialized constraints are valid for one
        specific configuration of the velocity diagram. Two possible configurations usually exist for wedge elements.
        All possible combinations of the constraints will then be automatically generated and considered during the
        optimization process.
    x0 : array-like
        Initial guess for the optimization variables.
    mode : str
        The failure mode of the mechanism. "T": translation; "RF": rotation about the foot
    params : dict
        Dictionary storing the configuration of the boundary value problem:
            * ALPHA: Wall inclination;
            * BETA: Inclination of the backfill
            * GAMMA: Soil unit weight
            * DELTA: Soil-wall interface friction
            * PHI: Soil friction angle
    h_soil : float
        Vertical height of the soil backfill right behind the wall.
    optimize_method : str
        Optimization method to use.

    Methods
    -------
    set_parameters(phi, delta, alpha, beta, gamma=None)
        Set the values for the mechanism parameters.
    set_parameter_by_name(name, value)
        Set the value of a specific parameter.
    plot_optimized_mech(ax=None, plot_wall=False, plot_backfill=False, **plot_kwargs)
        Plot the optimized mechanism configuration.
    draw_backfill(ax)
        Draw the backfill in the given axes.
    draw_wall(ax)
        Draw the wall in the given axes.
    plot_mech(x, ax=None, plot_wall=False, plot_backfill=False, **plot_kwargs)
        Plot the mechanism configuration for the parameters stored in the x array.
    external_energy(x)
        Calculate the external energy of the mechanism for the parameters stored in the x array.

    """

    def __init__(self):
        """
        Initializes an instance of the Mechanism class.

        Parameters:
            None

        Returns:
            None
        """
        self.elements = []
        self.element_config_plot = []
        self.bounds = []
        self.lconstr = None
        self.lconstr_spec = []
        self.optimize_result = None
        self.x0 = np.array([])
        self.mode = ""
        self.modes = ("T", "RF")
        self.params = dict()
        self.h_soil = 1.
        self.optimize_method = "trust-constr"
        self.optim_options = {
            "trust-constr": {},
            "SLSQP": {"ftol": 1e-7, "disp": False},
            "genetic": {"seed": 10,
                        "popsize": 15, "mutation": (0.2, 1)},
        }

        # phi, delta, alpha, beta
        self.set_parameters(30., 0., 0., 0., 1.)

    def set_parameters(self, phi, delta, alpha, beta, gamma=None):
        """
        Set the parameters of the mechanism.

        Parameters
        ----------
        phi : float
            Soil friction angle.
        delta : float
            Soil-wall interface friction.
        alpha : float
            Wall inclination.
        beta : float
            Inclination of the backfill.
        gamma : float, optional
            Unit soil weight. If not provided, it will be set based on the mode of the mechanism.
            If the mode is "T", gamma will be set to 2. If the mode is "RF", gamma will be set to 6.
            Otherwise, gamma will be set to 1. That way, the obtained result already corresponds to the load factor K,
            as for a rotation it would be: M=K*gamma*h**/6; and for a translation it would be: E=K*gamma*h**/2

        """
        if gamma is None:
            if self.mode == "T":
                gamma = 2.
            elif self.mode == "RF":
                gamma = 6.
            else:
                gamma = 1.
                warnings.warn("Failure mode {:s} is not yet implemented or it is misspelled. " +
                              "gamma has been set to unity.".format(self.mode))
        self.params = {PHI: phi, DELTA: delta, ALPHA: alpha, BETA: beta, GAMMA: gamma}

    def set_parameter_by_name(self, name: Parameters, value):
        """
        Set a single parameter by name.

        Parameters
        ----------
        name : Parameters
            The name of the parameter to be set.
            Should be one of the predefined Parameters values: ALPHA, BETA, GAMMA, DELTA, PHI.

        value : Union[float, np.ndarray]
            The value to be assigned to the parameter.

        """
        self.params[name] = value

    def plot_optimized_mech(self, ax=None, plot_wall=False, plot_backfill=False, **plot_kwargs):
        """
        Plot the optimized mechanism configuration.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            The axes on which to plot the mechanism. If not provided, a new figure and axes will be created.

        plot_wall : bool, optional
            Whether or not to plot the wall in the mechanism. Defaults to False.

        plot_backfill : bool, optional
            Whether or not to plot the backfill in the mechanism. Defaults to False.

        **plot_kwargs
            Additional keyword arguments to pass to the `plot` function of matplotlib. These will be used to customize
            the appearance of the mechanism plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the mechanism is plotted.

        """
        return self._plot_mech(self.optimized_config, ax, plot_wall, plot_backfill, **plot_kwargs)

    def _draw_backfill(self, ax):
        # Draw the backfill in the given axes.
        a = self.params[ALPHA]
        b = self.params[BETA]
        x_max = ax.get_xlim()[1]
        surf = (np.array([0, b]), np.array([0, x_max / np.cos(b)]))
        ax.plot(surf[1] * np.cos(surf[0]), surf[1] * np.sin(surf[0]), color='k', lw=0.7)
        ax.fill([0, self.h_soil * np.tan(a), x_max, x_max, 0],
                [0, -self.h_soil, -self.h_soil, x_max * np.tan(b), 0], color='.9')

    def _draw_wall(self, ax):
        # Draw the wall in the given axes.
        a = self.params[ALPHA]
        lw = self.h_soil / np.cos(a)
        rect = patches.Rectangle((0, -0.02 * lw), .06 * lw, 1.02 * lw)
        rot = mpl.transforms.Affine2D().rotate(np.pi + a)
        rect.set_transform(rot)
        verts = rect.get_verts()
        ax.plot(verts[:, 0], verts[:, 1], color='k', lw=0.5)
        ax.fill(verts[:, 0], verts[:, 1], color='.73')
        ax.annotate(xy=(np.average(verts[-3:-1, 0]), np.average(verts[-3:-1, 1])), xytext=(0, -2),
                    textcoords='offset points',
                    text="$\\mathcal{{\\alpha}}={:.0f}$\N{degree sign}".format(-np.rad2deg(a)), va='top', ha="center",
                    rotation=0)

    def _plot_mech(self, x: np.ndarray, ax=None, plot_wall=False, plot_backfill=False, **plot_kwargs):
        # Plot the mechanism configuration for the parameters stored in the x array.
        #
        #     x : np.ndarray
        #         The input array representing the mechanism's configuration.
        #     ax : matplotlib.axes.Axes, optional
        #         The axes object on which to plot the mechanism. If not provided, a new figure and axes object will be
        #          created.
        #     plot_wall : bool, optional
        #         Whether to plot the wall in the mechanism. Default is False.
        #     plot_backfill : bool, optional
        #         Whether to plot the backfill in the mechanism. Default is False.
        #     plot_kwargs : dict, optional
        #         Additional keyword arguments to pass to the `plot` method of each element in the mechanism.

        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if "color" not in plot_kwargs.keys():
            plot_kwargs["color"] = 'k'
        if "lw" not in plot_kwargs.keys():
            plot_kwargs["lw"] = 0.5

        self._update_mech_plot(x)

        for i, (element, config) in enumerate(zip(self.elements, self.element_config_plot)):
            if i > 0:
                if "label" in plot_kwargs:
                    del plot_kwargs["label"]
            if config is None:
                element.plot(ax, **plot_kwargs)
            else:
                element.plot(ax, config[0], config[1], **plot_kwargs)

        if plot_backfill:
            self._draw_backfill(ax)

        if plot_wall:
            self._draw_wall(ax)

        return ax

    def _external_energy(self, x: np.ndarray):
        # Calculate the external energy of the mechanism for the parameters stored in the x array.
        if type(x) is not np.ndarray:
            x = np.array(x)
        self._update_mech(x)
        energy = 0.
        for element in self.elements:
            try:
                energy -= element.external_energy(self.params[GAMMA])
            except InvalidConfiguration as e:
                print("x = " + np.array2string(x))
                raise InvalidConfiguration(e)
        return energy

    def _update_mech_plot(self, x: np.ndarray):
        # Update the mechanism configuration for plotting purposes.

        self._update_mech(x)

    def _optimize(self, repeated=0):
        # Solve the constrained optimization problem to find the most critical mechanism leading to the limit load.
        #
        # repeated : int, optional
        #     The number of times the optimization has already been repeated using different starting values.
        #     Default is 0.

        if type(self.x0) is not np.ndarray:
            self.x0 = np.array(self.x0)
        results = []

        self._optimizer(results)

        if len(results) > 0:
            vals = np.array([result.fun for result in results])
            self.optimize_result = results[np.argmax(vals == np.min(vals))]
        else:
            print("All configurations are inadmissible for " +
                  "phi = {:.1f}, delta = {:.1f}, alpha = {:.1f}, beta = {:.1f}".format(
                      np.rad2deg(self.params[PHI]), np.rad2deg(self.params[DELTA]),
                      np.rad2deg(self.params[ALPHA]),
                      np.rad2deg(self.params[BETA])))
            if repeated < 5:
                print("Trying another start value.")
                self._set_alternative_x0(repeated)
                self._optimize(repeated + 1)
            else:
                raise InvalidConfiguration()

    def _optimizer(self, results: list):
        # Run the optimization algorithm handling different possible situations.
        #
        # results: list
        #     A list which will contain the results of the optimization.

        if len(self.bounds) == 1:
            self._optimize_single_bound(results)
        else:
            self._optimize_multiple_bounds(results)

    def _optimize_single_bound(self, results: list):
        if self.lconstr is None:
            self._single_bound_no_general_constraints(results)
        else:
            self._single_bound_with_general_constraints(results)

    def _single_bound_no_general_constraints(self, results: list):
        if len(self.lconstr_spec) == 0:
            self._minimize_spec(results, self.bounds[0])
        else:
            for i, lconstr_spec in enumerate(self.lconstr_spec):
                self._minimize_spec(results, self.bounds[0], lconstr_spec, i)

    def _single_bound_with_general_constraints(self, results: list):
        if len(self.lconstr_spec) == 0:
            self._minimize_spec(results, self.bounds[0], [self.lconstr])
        else:
            for i, lconstr_spec in enumerate(self.lconstr_spec):
                if isinstance(lconstr_spec, Iterable):
                    constr = [*lconstr_spec, self.lconstr]
                else:
                    constr = [lconstr_spec, self.lconstr]
                self._minimize_spec(results, self.bounds[0], constr, i)

    def _optimize_multiple_bounds(self, results: list):
        if self.lconstr is None:
            self._multi_bounds_no_general_constraints(results)
        else:
            self._multi_bounds_with_general_constraints(results)

    def _multi_bounds_no_general_constraints(self, results: list):
        if len(self.lconstr_spec) == 0:
            for i, bound in enumerate(self.bounds):
                self._minimize_spec(results, bound, None, i)
        else:
            if len(self.lconstr_spec) != len(self.bounds):
                raise InconsistentConstraints(
                    "The dimension of the specialized linear constraints ({:d}) " +
                    "is not equal to the size of the bounds ({:d})".format(len(self.lconstr_spec),
                                                                           len(self.bounds)))
            for i, (bound, lconstr_spec) in enumerate(zip(self.bounds, self.lconstr_spec)):
                self._minimize_spec(results, bound, lconstr_spec, i)

    def _multi_bounds_with_general_constraints(self, results: list):
        if len(self.lconstr_spec) == 0:
            for i, bound in enumerate(self.bounds):
                self._minimize_spec(results, bound, [self.lconstr], i)
        else:
            if len(self.lconstr_spec) != len(self.bounds):
                raise InconsistentConstraints(
                    "The dimension of the specialized linear constraints ({:d}) " +
                    "is not equal to the size of the bounds ({:d})".format(len(self.lconstr_spec),
                                                                           len(self.bounds)))
            for i, (bound, lconstr_spec) in enumerate(zip(self.bounds, self.lconstr_spec)):
                if isinstance(lconstr_spec, Iterable):
                    constr = [*lconstr_spec, self.lconstr]
                else:
                    constr = [lconstr_spec, self.lconstr]
                self._minimize_spec(results, bound, constr, i)

    def _set_alternative_x0(self, repeated=0.):
        # Set an alternative initial guess for the optimization variables.
        # repeated : float, optional
        #     The number of times the optimization has been repeated for the same configuration. This parameter is only
        #     used in derived classes.

        # If not overridden in a child class, set an invalid x0: an admissible x0 will then be found by
        # _check_initial_value
        self.x0 = -np.ones(self.x0.shape)

    def _print_msg_catched_exception(self, i):
        # Print a message when a catched exception occurs during optimization.
        #
        # i : int
        #     The index of the current subconfiguration.

        print("The solver ended in an inadmissible configuration for " +
              "phi = {:.1f}, delta = {:.1f}, alpha = {:.1f}, beta = {:.1f}".format(
                  np.rad2deg(self.params[PHI]), np.rad2deg(self.params[DELTA]), np.rad2deg(self.params[ALPHA]),
                  np.rad2deg(self.params[BETA])) +
              "\n -> Subconfig {:d}/{:d}. ".format(i + 1, len(
                  self.lconstr_spec)) + "Other configurations could converge (in that case," +
              " you can ignore this message).")

    def _minimize_spec(self, results: list, bound: Bounds, lconstr: Union[Sequence[LinearConstraint], None] = None,
                       i: int = 1):
        # Perform the optimization for a specific configuration.
        #
        # results : list
        #     A list to store the results of the optimization.
        #
        # bound : Bounds
        #     The bounds for the optimization problem.
        #
        # lconstr : Union[Sequence[LinearConstraint], None], optional
        #     The linear constraints for the optimization problem. Defaults to None.
        #
        # i : int, optional
        #     The index of the current configuration. Defaults to 1.

        if self.optimize_method == "genetic":
            if lconstr is None:
                results.append(differential_evolution(self._external_energy, **self.optim_options[self.optimize_method],
                                                      bounds=bound))
            else:
                results.append(differential_evolution(self._external_energy, **self.optim_options[self.optimize_method],
                                                      bounds=bound, constraints=lconstr))
        else:
            x0 = np.copy(self.x0)
            if self._check_bounds_constr(bound, lconstr):
                x0 = self._check_initial_value(x0, bound, lconstr)
            else:
                x0[:] = np.nan
            if not np.isnan(x0).any():
                try:
                    if lconstr is None:
                        results.append(
                            minimize(self._external_energy, x0, args=(), method=self.optimize_method, bounds=bound,
                                     options=self.optim_options[self.optimize_method]))
                    else:
                        results.append(minimize(self._external_energy, x0, args=(), method=self.optimize_method,
                                                constraints=lconstr, bounds=bound,
                                                options=self.optim_options[self.optimize_method]))
                except (ValueError, InvalidConfiguration):
                    self._print_msg_catched_exception(i)
            else:
                self._print_msg_catched_exception(i)

    @abstractmethod
    def _update_mech(self, x: np.ndarray):
        # Update the mechanism configuration for plotting purposes.
        pass

    @staticmethod
    def _check_bounds_constr(bounds: Bounds, lconstr: Union[Sequence[LinearConstraint], None] = None):
        # Check if the bounds and constraints are valid.

        if np.greater(bounds.lb, bounds.ub).any():
            return False
        if lconstr is not None:
            for constr in lconstr:
                if np.greater(constr.lb, constr.ub).any():
                    return False
        return True

    @staticmethod
    def _check_initial_value(x0: np.ndarray, bounds: Bounds, lconstr: Union[Sequence[LinearConstraint], None] = None):
        # Check whether the initial guess satisfy the bounds and constraints. Update it (using brute force) otherwise.

        diff_lb = x0 - bounds.lb
        diff_ub = bounds.ub - x0
        mask = np.logical_or(diff_lb < 0, diff_ub < 0)
        x0[mask] = np.array(bounds.lb)[mask] + .1 * (-np.array(bounds.lb)[mask] + np.array(bounds.ub)[mask])

        if lconstr is not None:
            if len(lconstr) > 0:
                correct = False
                i = 0
                while not correct:
                    i += 1
                    local_ok = 0
                    j = 0
                    for lcon in lconstr:
                        j += 1
                        dotp = lcon.A.dot(x0)
                        if np.greater(dotp, lcon.ub).any() or np.less(dotp, lcon.lb).any():
                            x0 = np.random.uniform(np.array(bounds.lb), np.array(bounds.ub), x0.shape)
                        else:
                            local_ok += 1
                    correct = local_ok == j
                    if i == 500:
                        print("I'm having some troubles finding an admissible start value for the optimization. " +
                              "I haven't found any in {:d} guesses.\nLet me try additional".format(i) +
                              " 50 guesses before giving up")
                    if i >= 550:
                        x0[:] = np.nan
                        return x0
        return x0

    @property
    def resultant_force(self):
        return -self.optimize_result.fun

    @property
    def optimized_config(self):
        return self.optimize_result.x

    @property
    def ka(self):
        return self.kah / np.cos(self.params[ALPHA] + self.params[DELTA])

    @property
    def kah(self):
        a = self.params[ALPHA]
        d = self.params[DELTA]
        if self.mode == "T":
            factor = self.params[GAMMA] * self.h_soil ** 2. / 2.

        elif self.mode == "RF":
            factor = self.params[GAMMA] * self.h_soil ** 3. / 6. * np.cos(d) / (np.cos(a) * np.cos(a + d))
        else:
            raise UnavailableFailureMode(
                "Failure mode {:s} is not yet implemented (or is it just misspelled?).".format(self.mode))
        return self.resultant_force / factor

    @property
    def kan(self):
        return self.ka * np.cos(self.params[DELTA])
