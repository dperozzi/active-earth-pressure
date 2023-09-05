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

class UnavailableFailureMode(Exception):
    """
    An exception class for representing an unavailable failure mode.
    """
    pass


class InconsistentConstraints(Exception):
    """
    Exception raised when there are inconsistent constraints.
    """
    pass


class InvalidConfiguration(Exception):
    """
    An exception that is raised when an invalid configuration is encountered.
    """
    pass
