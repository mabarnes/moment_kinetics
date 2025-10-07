Jacobian matrix calculations
============================

Functions are provided for constructing Jacobian matrices (primarily for use in
constructing preconditioners) in [`moment_kinetics.jacobian_matrices`](@ref).

To illustrate the process and the provided functions, consider a simplified
example equation
```math
\frac{\partial f}{\partial t} + \sqrt{p} \frac{\partial f}{\partial z} = 0
```
where $p$ is another time-evolving variable with a separate evolution equation
($p$'s evolution equation is ignored here for simplicity).

The discrete version of this equation (assuming backward-Euler implicit
timestepping) is
```math
\begin{align}
\frac{f_i^{n+1} - f_i^n}{\Delta t} + \sqrt{p_i^{n+1}} \sum_j D_{ij} f_j^{n+1} &= 0 \\
R_{f,i}[f,p] \equiv f_i^{n+1} - f_i^n + \Delta t \sqrt{p_i^{n+1}} \sum_j D_{ij} f_j^{n+1} &= 0
\end{align}
```
where $D_{ij}$ is a matrix representing the discretised derivative.
This is part of a non-linear system of equations (with the corresponding
equation for $p$) that we solve with the JFNK method to take implicit
timesteps.

The Jacobian is the linearisation of $R_i$ around some current value/guess
$f_{(0)},p_{(0)}$. We use a 'lagged Jacobian' as a preconditioner, which means
calculating the Jacobian matrix at some time point, then using that matrix as
the preconditioner for several following timesteps.

The rows of the Jacobian matrix corresponding to the evolution equation for $f$
are given by
```math
\frac{\delta R_{f,i}}{\delta f_j}[f_{(0)},p_{(0)}] = \delta_{ij} + \Delta t \sqrt{p_{(0),i}} D_{ij}
```
for the columns corresponding to $f$, and
```math
\frac{\delta R_{f,i}}{\delta p_j}[f_{(0)},p_{(0)}] = \delta_{ij} \frac{1}{2 \sqrt{p_{(0),i}}} \left(\frac{\partial f_{(0)}}{\partial z}\right)_i
```
for the columns corresponding to $p$.

These expressions can be constructed by combining some simpler building blocks,
represented in the code by instances of the
[`moment_kinetics.jacobian_matrices.EquationTerm`](@ref) struct.

 $f$ and $p$ themselves are represented by 'simple' objects for which we know
the functional derivatives
```math
\frac{\delta f_i}{\delta f_j} = \delta_{ij},\quad \frac{\delta p_i}{\delta p_j} = \delta_{ij}
```
Derivatives of variables similarly are represented by objects for which we
again know the functional derivatives, e.g.
```math
\frac{\delta (\partial_z f)_i}{\delta f_j} = D_{ij}
```

Sums or products of other terms are represented by `EquationTerm` objects that
contain other `EquationTerm` 'sub terms' that we can combine by the chain rule, etc.
```math
\begin{align}
\frac{\delta (a + b)_i}{\delta f_j} = \frac{\delta a_i}{\delta f_j} + \frac{\delta b_i}{\delta f_j} \\
\frac{\delta (a b)_i}{\delta f_j} = b_i \frac{\delta a_i}{\delta f_j} + a_i \frac{\delta b_i}{\delta f_j}
\end{align}
```
Powers can also be handled
```math
\frac{\delta (a^p)_i}{\delta f_j} = p a^{p-1} \frac{\delta a_i}{\delta f_j}
```

The electron kinetic equation also needs an integral over velocity space for
$\partial q_\parallel \partial z$ coeffients. For example something like
```math
\begin{align}
X(z) &= \int P(v) f(z,v) dv \\
X_{i_z}  &= \sum_{j_v} P_{j_v} f_{i_z,j_v} w_{j_v} \\
\frac{\delta X_{i_z}}{\delta f_{j_z,j_v}} &= \delta_{i_z,j_z} P_{j_v} w_{j_v} \\
\end{align}
```
where $w_j$ are the weights used to numerically calculate the integral from
function values on grid points. For a derivative of an integral
```math
\begin{align}
X(z) &= \frac{\partial}{\partial z} \int P(v) f(z,v) dv \\
X(z) &= \int P(v) \frac{\partial f(z,v)}{\partial z} dv \\
X_{i_z}  &= \sum_{j_v} P_{j_v} \sum_{j_z} D_{i_z,j_z} f_{j_z,j_v} w_{j_v} \\
\frac{\delta X_{i_z}}{\delta f_{j_z,j_v}} &= P_{j_v} D_{i_z,j_z} w_{j_v} \\
\end{align}
```

By representing the equation for $R_{f,i}$ as a tree of `EquationTerm` objects,
we can travel down the tree until we get to a 'leaf' node, and add its
contribution to some element (or multiple elements in different columns for a
derivative or integral) of the Jacobian matrix. This is done by
[`moment_kinetics.jacobian_matrices.add_term_to_Jacobian!`](@ref), which calls
recursively
[`moment_kinetics.jacobian_matrices.add_term_to_Jacobian_row!`](@ref).

This setup allows a separation of different parts of the Jacobian construction
logic. For example, for the Jacobian for the kinetic electron solve is
constructed step by step in
[`moment_kinetics.electron_kinetic_equation.fill_electron_kinetic_equation_Jacobian!`](@ref):
1. The state vector variables, and basic integrals and derivatives are defined
   in
   [`moment_kinetics.electron_kinetic_equation.get_electron_sub_terms`](@ref).
2. The 'sub terms' are assembled into an `EquationTerm` object representing the
   kinetic and pressure equations in
   [`moment_kinetics.electron_kinetic_equation.get_all_electron_terms`](@ref),
   which calls other functions that get the contribution from each individual
   term, e.g. parallel streaming ('z advection'), etc.
3. The contributions of all the terms are added to a Jacobian matrix by calling
   [`moment_kinetics.jacobian_matrices.add_term_to_Jacobian!`](@ref).

There are other types of contribution that are not naturally included within
the `EquationTerm` setup. For example, the wall boundary condition is not
simply expressed as an integro-differential operator, so is handled by an
ad-hoc function
[`moment_kinetics.electron_kinetic_equation.add_wall_boundary_condition_to_Jacobian!`](@ref).
