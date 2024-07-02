# Lid-driven cavity problem
Recreating the lid-driven cavity problem described in *Hesthaven, J. S. (2018). Nonintrusive reduced order modeling of nonlinear problems using neural networks. Journal of Computational Physics, 395, 499-509*.

The lid-driven cavity flow problem is concerned with solving the steady-state, incompressible Navier-Stokes equations, within a parallelogram-shaped cavity. The motion is instigated by the movement of one wall, referred to as the “lid".

$$ -\nu \nabla^2u + u \nabla u + \frac{\nabla p}{\rho} = g$$

$$ \nabla u = 0$$

The problem has three geometric parameters: $a$, $b$, $θ$, and two physical parameters: density, $\rho$, and viscosity, $\nu$. In our experiments we keep $\rho$ and $\mu$ fixed at 1, and vary geometrical parameters.