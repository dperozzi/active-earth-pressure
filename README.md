# Determining the Active Earth Pressure on Retaining Walls: A Limit Analysis-Based Tool

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Important note](#important-note)
6. [License and Copyright](#license-and-copyright)

## Introduction

This code serves as a supplementary resource for the paper titled "Limit-state solutions for the active earth
pressure behind walls rotating about the base", authored by D. Perozzi and A. M. Puzrin, and submitted to Géotechnique.
The project focuses on solving the active earth pressure problem on a wall rotating about its foot
(although a horizontal translation can also be considered) using the kinematic solution of limit analysis.
The code aims to help researchers and practitioners understand and apply the methodologiesdescribed in the paper
more effectively. Also, this code allows to calculate an approximate static solution (based on Lancellotta's original
solution) to the same problem.

The code is modular, and additional mechanisms could be added in [mechanisms.py](earth_pressure_la/mechanisms.py), by
combining the base elements (triangular block and logarithmic spiral sector) formulated in
[elements.py](earth_pressure_la/elements.py).
These elements can also be used to solve any other limit analysis problem, such as the bearing capacity of strip
foundations or the passive earth pressure.

For the latest updates and possible newer versions of this code, please visit the
[active-earth-pressure repository on GitHub](https://github.com/dperozzi/active-earth-pressure.git).

## Prerequisites

Before you begin, ensure you have met the following requirements:

### Option 1: Python Environment

- Python 3.x installed. You can download it from [python.org](https://www.python.org/downloads/).

### Option 2: Docker

- Docker installed. You can download it from [docker.com](https://www.docker.com/products/docker-desktop).

Choose one of the above options to proceed with the installation.

## Installation

Step-by-step guide on setting up your project:

```bash
git clone https://github.com/dperozzi/active-earth-pressure.git

cd active-earth-pressure
```

Then, if you have a working python environment:

```bash
# Install dependencies
pip install .
```

If you have Docker:

```bash
docker build -t earth_pressure .

docker run -p 8888:8888 earth_pressure
```

## Usage

**Starting the Jupyter Lab Session:**

Upon starting the Docker container, a Jupyter Lab session will be initiated automatically. To access it, refer to the
console output for the access link provided when you run the container. This link will direct you to the Jupyter Lab
interface in your web browser.

If you are not using docker, just open a terminal window and start jupyter by typing the command

```bash
jupyter lab
```

For an example usage, refer to the files [calculate_earth_pressure.ipynb](calculate_earth_pressure.ipynb) and
[earth_pressure_paper.ipynb](earth_pressure_paper.ipynb)

To calculate the earth pressure coefficients, you can open the
file [calculate_earth_pressure.ipynb](calculate_earth_pressure.ipynb)
in jupyter lab.

## Important note

The code follows the sign conventions and parameter definitions set forth in the paper by Perozzi and Puzrin (2023),
with *one notable exception* concerning the sign convention for the wall inclination, denoted as alpha.
In the modules of this code, opposite signs are used for alpha compared to the paper's convention.
This discrepancy has been addressed and corrected in the
notebooks [calculate_earth_pressure.ipynb](calculate_earth_pressure.ipynb)
and [earth_pressure_paper.ipynb](earth_pressure_paper.ipynb), where the sign convention aligns with that used in the
paper.

## License and Copyright

Copyright (c) 2023. ETH Zurich, David Perozzi; D-BAUG; Institute for Geotechnical Engineering; Chair of Geomechanics and
Geosystems Engineering.

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
