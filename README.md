# FESTIM

[![CircleCI](https://circleci.com/gh/RemDelaporteMathurin/FESTIM.svg?style=svg&circle-token=ecc5a4a8c75955af6c238d255465bc04dfaaaf8e)](https://circleci.com/gh/RemDelaporteMathurin/FESTIM)
[![codecov](https://codecov.io/gh/RemDelaporteMathurin/FESTIM/branch/master/graph/badge.svg?token=AK3A9CV2D3)](https://codecov.io/gh/RemDelaporteMathurin/FESTIM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
![GitHub stars](https://img.shields.io/github/stars/RemDelaporteMathurin/FESTIM.svg?logo=github&label=Stars&logoColor=white)
![GitHub forks](https://img.shields.io/github/forks/RemDelaporteMathurin/FESTIM.svg?logo=github&label=Forks&logoColor=white)

FESTIM (Finite Elements Simulation of Tritium in Materials) is a tool for modeling hydrogen transport in materials. 
FESTIM simulates the diffusion and trapping of hydrogen, coupled to heat transfer.


The following features are included:
- Mesh import from XDMF files
- **Adaptive stepsize**
- **Temperature** from solving transient/stationnary heat equation
- Multiple intrinsic/extrinsic traps with **non-homogeneous density distribution**
- Wide range of built-in boundary conditions (Sievert's law, recombination flux, user-defined expression...)
- **Derived quantities** computation (surface fluxes, volume integrations, extrema over domains, mean values over domains...)
- Soret effect
- ...

FESTIM spatially discretises the PDEs using the Finite Element Methods and heavily relies on [FEniCS](https://fenicsproject.org).

## Installation

FESTIM can be installed via pip

    pip install FESTIM

FESTIM requires FEniCS to run.
The FEniCS project provides a [Docker image](https://hub.docker.com/r/fenicsproject/stable/) with FEniCS and its dependencies (python3, UFL, DOLFIN, numpy, sympy...)  already installed. See their ["FEniCS in Docker" manual](https://fenics.readthedocs.io/projects/containers/en/latest/).

For Windows users:

    docker run -ti -v ${PWD}:/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

For Linux users:

    docker run -ti -v $(pwd):/home/fenics/shared --name fenics quay.io/fenicsproject/stable:latest

Run the tests:

    pytest-3 test/


## Visualisation
FESTIM results are exported to .csv, .txt or XDMF. The latter can then be opened in visualisation tools like [ParaView](https://www.paraview.org/) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).
<p align="center">
  <img alt="performance" src="https://user-images.githubusercontent.com/40028739/69346147-9abb6980-0c72-11ea-80e7-9c0a76659268.png" width="40%"> <img alt="performance" src="https://user-images.githubusercontent.com/40028739/69346752-9d6a8e80-0c73-11ea-96c1-27b6104eb9ff.png" width="40%">
</p>

## References
- R. Delaporte-Mathurin, _et al._, _Finite Element Analysis of Hydrogen Retention in ITER Plasma Facing Components using FESTIM_. Nuclear Materials and Energy 21: 100709. DOI: [10.1016/j.nme.2019.100709](https://doi.org/10.1016/j.nme.2019.100709).

- R. Delaporte-Mathurin, _et al._, _Parametric Optimisation Based on TDS Experiments for Rapid and Efficient Identification of Hydrogen Transport Materials Properties_. Nuclear Materials and Energy, 26 mars 2021, 100984. https://doi.org/10.1016/j.nme.2021.100984.
