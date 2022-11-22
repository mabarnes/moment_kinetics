Constraints on normalized distribution function
===============================================

> Note: Equation references give the Excalibur/Neptune report number and equation number, e.g. (TN-04;1) is equation (1) from report TN-04.pdf.

Constraints
-----------

The normalized particle distribution function that is evolved when using the moment-kinetic approach has to satisfy integral constraints related to particle number, momentum and energy conservation (TN-04;70-72)

```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dw_{\|}\tilde{g}_{s} & =1\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}\tilde{g}_{s} & =0\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{2}\tilde{g}_{s} & =\frac{1}{2}
\end{align}
```

Old algorithm
-------------

The algorithm described in TN-04 used the distribution function from the previous time step and also made use of a symmetrized distribution function $\tilde{g}_{E}(w_{\|})=\frac{1}{2}\left(\tilde{g}(w_{\|})+\tilde{g}(-w_{\|})\right)$. These choices caused problems when in combination with the boundary conditions as: applying the boundary condition at the new timestep (with an updated $\tilde{u}_{\|}$) to the old $\tilde{g}_{s}$ could mean that it no longer satisfied the moment constraints (e.g. if a grid point that was previously non-zero is now set to zero by the ion sheath boundary condition); the symmetrized $\tilde{g}_{E}$ will be non-zero at places where the boundary condition forces $\tilde{g}_{s}$ to be zero. It is possible to extend the algorithm to allow the constraints to be enforced using only the initial guess of the distribution function at the new timestep, as described below.

Current algorithm
-----------------

After the time advance updates the distribution function, it will in general not obey the constraints, but the errors will be small, with the size depending on the accuracy of the spatial and temporal discretizations. We can take this updated value $\hat{g}_{s}$ as an initial guess, to be corrected to give the actual updated value $\tilde{g}_{s}$, which does obey the constraints to machine precision.

We define the corrected distribution function as

```math
\begin{align}
  \tilde{g}_{s}=A\hat{g}_{s}+Bw_{\|}\hat{g}_{s}+Cw_{\|}^{2}\hat{g}_{s}
\end{align}
```
and define the moments of $\hat{g}_{s}$

```math
\begin{align}
  I_{n}=\frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{n}\hat{g}_{s}.
\end{align}
```

Then the moments of $\tilde{g}_{s}$ are

```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dw_{\|}\tilde{g}_{s}=1 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(A\hat{g}_{s}+Bw_{\|}\hat{g}_{s}+Cw_{\|}^{2}\hat{g}_{s}\right)=AI_{0}+BI_{1}+CI_{2}\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}\tilde{g}_{s}=0 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(Aw_{\|}\hat{g}_{s}+Bw_{\|}^{2}\hat{g}_{s}+Cw_{\|}^{3}\hat{g}_{s}\right)=AI_{1}+BI_{2}+CI_{3}\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{2}\tilde{g}_{s}=\frac{1}{2} & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(Aw_{\|}^{2}\hat{g}_{s}+Bw_{\|}^{3}\hat{g}_{s}+Cw_{\|}^{4}\hat{g}_{s}\right)=AI_{2}+BI_{3}+CI_{4}.
\end{align}
```

Solving the simultaneous equations for $A$, $B$, $C$ gives

```math
\begin{align}
  C & =\frac{\frac{1}{2}-AI_{2}-BI_{3}}{I_{4}}\\
  B & =\frac{\left(I_{2}I_{3}-I_{1}I_{4}\right)A-\frac{I_{3}}{2}}{I_{2}I_{4}-I_{3}^{2}}\\
  A & =\frac{I_{2}I_{4}-\frac{I_{2}^{2}}{2}+I_{3}\left(\frac{I_{1}}{2}-I_{3}\right)}{I_{0}\left(I_{2}I_{4}-I_{3}^{2}\right)+I_{1}\left(I_{2}I_{3}-I_{1}I_{4}\right)+I_{2}\left(I_{1}I_{3}-I_{2}^{2}\right)}.
\end{align}
```

Note that there is no guarantee that $\tilde{g}_{s}$ is $\geq0$ even if $\hat{g}_{s}\geq0$, although if the violations of the integral constraints are small, it should be true that $A\approx1$ while $B$ and $C$ are small.
