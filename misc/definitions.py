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

from enum import IntEnum, unique, auto

import numpy as np
from scipy.optimize import Bounds as BoundsOriginal
from scipy.optimize import LinearConstraint as LinearConstraintOriginal


# Global variables
@unique
class Parameters(IntEnum):
    """
    Represents an enumeration of parameters.

    Enum Values:
    ALPHA : int
        Represents the wall inclination.
    BETA : int
        Represents the backfill inclination.
    GAMMA : int
        Represents the soil unit weight.
    DELTA : int
        Represents the interface friction angle.
    PHI : int
        Represents the soil friction angle.
    """
    ALPHA = 0
    BETA = auto()
    GAMMA = auto()
    DELTA = auto()
    PHI = auto()


# Define them as global variables
ALPHA = Parameters.ALPHA
BETA = Parameters.BETA
GAMMA = Parameters.GAMMA
DELTA = Parameters.DELTA
PHI = Parameters.PHI


@unique
class Mechanisms(IntEnum):
    """
    Represents an enumeration of mechanisms.

    Enum Values:
    ONE_WEDGE : int
        Represents the "One Wedge" mechanism.
    TWO_WEDGES : int
        Represents the "Two Wedges" mechanism.
    EXT_ONE_WEDGE : int
        Represents the "Extended One Wedge" mechanism.
    LOG_SANDWICH : int
        Represents the "Log Sandwich" mechanism.
    """
    ONE_WEDGE = 0
    TWO_WEDGES = auto()
    EXT_ONE_WEDGE = auto()
    LOG_SANDWICH = auto()


# Define them as global variables
ONE_WEDGE = Mechanisms.ONE_WEDGE
TWO_WEDGES = Mechanisms.TWO_WEDGES
EXT_ONE_WEDGE = Mechanisms.EXT_ONE_WEDGE
LOG_SANDWICH = Mechanisms.LOG_SANDWICH


class Bounds(BoundsOriginal):
    """
    This class extends the Bounds class from the scipy.optimize module. It represents the bounds on the variables
    for optimization problems.

    Attributes
    ----------
    lb : array_like
        The lower bound for each variable.
    ub : array_like
        The upper bound for each variable.
    keep_feasible : bool, optional
        Flag to indicate whether the bounds should be kept feasible throughout iterations. Defaults to False.

    """

    def __init__(self, lb, ub, keep_feasible=False):
        super().__init__(lb, ub, keep_feasible)
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)


class LinearConstraint(LinearConstraintOriginal):
    """
    This class extends the LinearConstraint class from the scipy.optimize module. It represents the constraints on
    the variables for optimization problems.

    Attributes
    ----------
    A : array_like
        Coefficient matrix of the linear constraint.
    lb : dense array_like
        Lower bounds of the linear constraint.
    ub : dense array_like
        Upper bounds of the linear constraint.
    keep_feasible : bool, optional
        Flag to indicate whether the bounds should be kept feasible throughout iterations. Defaults to False.

    """

    def __init__(self, A, lb, ub, keep_feasible=False):
        super().__init__(A, lb, ub, keep_feasible)
        self.A = np.array(self.A)
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
