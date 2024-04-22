# Bacherlor Final Project - Lászlo Roovers - 2024
## Comparison of ML-architectures for short-term prediction in 2D turbulence

The goal of the project is to compare how well different machine learning (ML) architectures can make short-term predictions of solution to the Navier-Stokes equations. We consider the vorticity formulation, as given by:

$$\frac{\partial \omega}{\partial t} + \mathcal{J}(\omega, \psi) = \nu \nabla^2 \omega + \mu(f - \omega),$$

$$\nabla^2 \psi= \omega,$$

where $\omega$ denotes the vorticity, $\psi$ denotes the stream function and, $J$ is the advection operator

$$
    \mathcal{J}(\omega, \psi) = \frac{\partial \psi}{\partial x} \frac{\partial \omega}{\partial y} - \frac{\partial \psi}{\partial y} \frac{\partial \omega}{\partial x}.
$$

The architectures considered throughout the project are:
- Convolutional Neural Networks
- Residual Neural Networks
- U-Nets
