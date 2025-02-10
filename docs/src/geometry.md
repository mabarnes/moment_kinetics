Magnetic Geometry
=================

We take the magnetic field $\mathbf{B}$ to have the form 
```math
\begin{equation}
\mathbf{B} = B_z \hat{\mathbf{z}} + B_\zeta \hat{\mathbf{\zeta}},
\end{equation}
```
with $B_\zeta = B(r,z) b_\zeta$, $B_z = B(r,z) b_z$ and $b_z$ and $b_\zeta$ the direction cosines of the magnetic field vector.
Here the basis vectors are those of cylindrical geometry $(r,z,\zeta)$, i.e.,
$\hat{\mathbf{r}} = \nabla r $, $\hat{\mathbf{z}} = \nabla z$, 
and $\hat{\mathbf{\zeta}} = r \nabla \zeta$. The unit vectors $\hat{\mathbf{r}}$, $\hat{\mathbf{z}}$, and $\hat{\mathbf{\zeta}}$
form a right-handed orthonormal basis.

Supported options
-----------------

To choose the type of geometry, set the value of "option" in the geometry namelist. The namelist will have the following appearance in the TOML file.
```
[geometry]
option="constant-helical" # ( or "1D-mirror" )
pitch = 1.0
rhostar = 1.0
DeltaB = 0.0
```
If `rhostar` is not set then it is computed from reference parameters.

### [geometry] option = "constant-helical"

Here $b_\zeta = \sqrt{1 - b_z^2}$ is a constant, $b_z$ is a constant input parameter ("pitch") and $B$ is taken to be 1 with respect to the reference value $B_{\rm ref}$.

### [geometry] option = "1D-mirror"

Here $b_\zeta = \sqrt{1 - b_z^2}$ is a constant, $b_z$ is a constant input parameter ("pitch") and $B = B(z)$ is taken to be 
the function 
```math
\begin{equation}
\frac{B(z)}{B_{\rm ref}} = 
    1 + \Delta B \left( 2\left(\frac{2z}{L_z}\right)^2 - \left(\frac{2z}{L_z}\right)^4\right)
\end{equation}
```
where $\Delta B $ is an input parameter ("DeltaB") that must satisfy $\Delta B > -1$.
Recalling that the coordinate $z$ runs from 
$z = -L_z/2$ to $L_z/2$,
if $\Delta B > 0$ than the field represents a magnetic mirror which traps particles,
whereas if $\Delta B < 0$ then the magnetic field accelerates particles
by the mirror force as they approach the wall.
Note that this field does not satisfy $\nabla \cdot \mathbf{B} = 0$, and is
only used to test the implementation of the magnetic mirror terms. 2D simulations with a radial domain and$\mathbf{E}\times\mathbf{B}$ drifts are supported in the "1D-mirror"
geometry option.

Geometric coefficients
----------------------
Here, we write the geometric coefficients appearing in the characteristic equations
explicitly.

The $z$ component of the $\mathbf{E}\times\mathbf{B}$ drift is given by
```math
\begin{equation} \frac{\mathbf{E}\times\mathbf{B}}{B^2} \cdot \nabla z = \frac{E_r B_\zeta}{B^2} \nabla r \times \hat{\mathbf{\zeta}} \cdot \nabla z 
= - J \frac{E_r B_\zeta}{B^2},
\end{equation}
```
where we have defined $J = r \nabla r \times \nabla z \cdot \nabla \zeta$.
Note that $J$ is dimensionless.
The $r$ component of the $\mathbf{E}\times\mathbf{B}$ drift is given by
```math
\begin{equation} \frac{\mathbf{E}\times\mathbf{B}}{B^2} \cdot \nabla r = \frac{E_z B_\zeta}{B^2} \nabla z \times \hat{\mathbf{\zeta}} \cdot \nabla r 
=  J \frac{E_z B_\zeta}{B^2}.
\end{equation}
```
Due to the axisymmetry of the system, the differential operator
 $\mathbf{b} \cdot \nabla (\cdot)  = b_z \partial {(\cdot)}{\partial z}$,
 and the convective derivative
```math
\begin{equation}
\frac{d B}{d t} = \frac{d z}{d t} \frac{\partial B}{ \partial z} + \frac{dr}{dt}\frac{\partial B}{\partial r}.
\end{equation}
```
