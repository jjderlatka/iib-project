# IIB Project

## Introduction
Through parametric PDEs, we can formulate problems with variable configurations, where material properties, boundary conditions, and geometry can be defined in terms of parameters. 
Numerical approaches to such problems require re-evaluation at parameters change, making them infeasible for real-time or multi-query applications, like control systems or engineering design optimisation.
The POD-ANN method investigated, in offline phase, builds a manifold of solutions of the parametric problem under variation of the parameters, which can be quickly queried in the online phase. It was first introduced by Hesthaven, J. S. in 2018.

## 🎯 Project overview
The method was implemented integrating open-source libraries: FEniCSx was used to calculate FEM ground truth solutions, MDFEniCSx for mesh deformation, RBniCSx for POD, and PyTorch based DLRBniCSx for neural network implementation.
The offline stage features parallelisation both in dataset generation and neural network training, improving the efficiency and promoting scaling. The implementation is self-contained, allowing anyone to experiment with this method. It is hoped that this work will serve as a useful reference in future related research.

## 📄 Report
Full report on the project can be found in the root directory of the repo.

## ⬇️ Installation
Most dependencies required for this project are easily obtained by using the [docker image](https://github.com/jorgensd/dolfinx-tutorial/pkgs/container/dolfinx-tutorial). The missing ones, which have to be cloned manually are
[RBniCSx](https://github.com/RBniCS/RBniCSx), 
[MDFEniCSx](https://github.com/niravshah241/MDFEniCSx)
and [DLRBniCSx](https://github.com/Wells-Group/dlrbnicsx).

## 🚀 Running the code
Instructions on running the code can be found in each demo's folder.