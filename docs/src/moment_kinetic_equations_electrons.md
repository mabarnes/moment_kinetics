Moment kinetic equations for electrons
======================================

Moment equations
----------------

Quasineutrality gives
```math
\begin{align}
n_e &= n_i
\end{align}
```
while the continuity equation is
```math
\begin{align}
\frac{\partial n_e}{\partial t}
+ \frac{\partial}{\partial z}\left( n_e u_{e\parallel} \right)
&= n_e n_n R_\mathrm{ioniz} + S_{e,n}
\end{align}
```
Subtracting from ion continuity and using quasineutrality
```math
\begin{align}
\frac{\partial}{\partial z}\left(n_e (u_{i\parallel} - u_{e\parallel}) \right)
&= S_{i,n} - S_{e,n}
\end{align}
```
giving an equation for $u_{e\parallel}$. Assuming that the sources do not add
any charge $S_{i,n} = S_{e,n}$ then implies that
$u_{e\parallel} = u_{i\parallel} + u_0$ assuming in addition (at least for the
moment) that there is no current through the sheath sets $u_0 = 0$ so that
```math
\begin{align}
u_{e\parallel} &= u_{i\parallel}
\end{align}
```

In the parallel momentum equation, the inertial terms can be neglected as they
are $O(m_e/m_i)$, leaving
```math
\begin{align}
0 &= -\frac{\partial p_{e\parallel}}{\partial z}
     + e n_e \frac{\partial \phi}{\partial z}
     + F_{ei\parallel}
     + m_e \int v_\parallel C_{en} d^3 v
     + S_{e,\mathrm{mom}}
\end{align}
```
where $C_{en}$ represents elastic electron-neutral collisions (for collisions
see [Electron collisions](@ref)), which gives an equation for
$E_\parallel = -\partial \phi/\partial z$
```math
\begin{align}
e E_\parallel &= -\frac{1}{n_e} \frac{\partial p_{e\parallel}}{\partial z}
                 + \frac{F_{ei\parallel}}{n_e}
                 + \frac{m_e}{n_e} \int v_\parallel C_{en} d^3 v
                 + \frac{1}{n_e} S_{e,\mathrm{mom}} \\
e E_\parallel &= -\frac{1}{n_e} \frac{\partial p_{e\parallel}}{\partial z}
                 + m_e \nu_{ei} \left( u_{i\parallel} - u_{e\parallel} \right)
                 + \frac{1}{n_e} S_{e,\mathrm{mom}} \\
\end{align}
```
where the second line assumes the current Krook collision operator and neglects
electron-neutral elastic collisions.

The energy equation is similar to [the ion one](@ref "Ion moment equations")
```math
\begin{align}
&\frac{3}{2} \frac{\partial p_e}{\partial t}
+ \frac{\partial q_{e\parallel}}{\partial z}
+ p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
+ \frac{3}{2} u_{e\parallel} \frac{\partial p_e}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + \int \frac{1}{2} m_e |\boldsymbol{v} - u_{e\parallel} \hat{\boldsymbol{z}}|^2 C_{ei} d^3 v
        + \int \frac{1}{2} m_e |\boldsymbol{v} - u_{e\parallel} \hat{\boldsymbol{z}}|^2 C_{en} d^3 v \nonumber \\
&\qquad + \frac{3}{2} S_{e,p} \\

&\frac{3}{2} \frac{\partial p_e}{\partial t}
+ \frac{\partial q_{e\parallel}}{\partial z}
+ p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
+ \frac{3}{2} u_{e\parallel} \frac{\partial p_e}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + 3 n_e \frac{m_e}{m_i} \nu_{ei} \left( T_i - T_e \right)
        + m_e n_e \nu_{ei} \left( u_{i\parallel} - u_{e_\parallel} \right)^2 \nonumber \\
&\qquad + \frac{3}{2} S_{e,p} \\
\end{align}
```
To get the second version we treat the collision operators as described in
[Electron collisions](@ref).

Electron collisions
-------------------

Currently electron-neutral elastic collisions $C_{en}$ are not implemented.

Electron-electron collisions use a Krook operator
```math
\begin{align}
C_{K,ee} &= -\nu_{ee}(n_e,T_e) \left(f_e - \frac{n_e}{\pi^{3/2} v_{Te}^3} \exp\left(-\frac{|\boldsymbol{v} - u_{e\parallel}\hat{\boldsymbol{z}}|^2}{v_{Te}^2}\right) \right)
\end{align}
```
with the electron-electron collision frequency
```math
\begin{align}
\nu_{ee}(n_e,T_e) = \frac{n_e e^4 \log\Lambda_{ee}}{4 \pi \epsilon_0^2 m_e^2 v_{Te}^3}
\end{align}
```

Electron-ion collisions also use a Krook operator
```math
\begin{align}
C_{K,ei} &= -\nu_{ei}(n_e,T_e) \left(f_e - \frac{n_e}{\pi^{3/2} v_{Te}^3} \exp\left(-\frac{|\boldsymbol{v} - u_{i\parallel}\hat{\boldsymbol{z}}|^2}{v_{Te}^2}\right) \right)
\end{align}
```
with the electron-ion collision frequency
```math
\begin{align}
\nu_{ei}(n_e,T_e) = \frac{n_e e^4 \log\Lambda_{ei}}{4 \pi \epsilon_0^2 m_e^2 v_{Te}^3}
\end{align}
```

For the Krook operator, the friction is
```math
\begin{align}
F_{ei\parallel} &= -m_e \int v_\parallel C_{K,ei} d^3 v \nonumber \\
                &= m_e n_e \nu_{ei} \left( u_{i\parallel} - u_{e\parallel} \right)
\end{align}
```

Energy exchange with ions is kept assuming the distributions are Maxwellian to
allow different temperatures (this form comes from the Fokker-Planck collision
operator assuming Maxwellian distributions)
```math
\begin{align}
\int \frac{1}{2} m_i |\boldsymbol{v} - u_{i\parallel} \hat{\boldsymbol{z}}|^2 C_{ie}[f_i,f_e] d^3 v
&\approx 3 \frac{n_e m_e \nu_{ei}}{m_i} (T_e - T_i)
\end{align}
```
in the ion energy equation. In the electron energy equation, need the
conversion
```math
\begin{align}
\int \frac{1}{2} m_e |\boldsymbol{v} - u_{e\parallel} \hat{\boldsymbol{z}}|^2 C_{ei}[f_e,f_i] d^3 v
&= \int \left(\frac{1}{2} m_e v^2 - m_e u_{e\parallel} v_\parallel + \underbrace{\frac{1}{2} m_e u_{e\parallel}^2}_{=0\text{ particle conservation}} \right) C_ei d^3 v \nonumber \\
&= -\int \frac{1}{2} m_i v^2 C_{ie}
   - m_e u_{e\parallel} \underbrace{\int v_\parallel C_{ei} d^3 v}_{F_{ei\parallel}/m_e} \nonumber \\
&= -\int \frac{1}{2} m_i |\boldsymbol{v} - u_{i\parallel} \hat{\boldsymbol{z}}|^2 C_ie d^3 v
   - \underbrace{\int m_i u_{i\parallel} v_\parallel C_{ie} d^3 v}_{\text{momentum conservation } = u_{i\parallel} (-F_{ie\parallel}) = u_{i\parallel} F_{ei\parallel}}
   - F_{ei\parallel} u_{e\parallel} \nonumber \\
&= 3 n_e \frac{m_e}{m_i} \nu_{ei} \left(T_i - T_e \right) + F_{ei\parallel} \left(u_{i\parallel} - u_{e\parallel} \right) \\
\end{align}
```

1D2V kinetic equation
---------------------

For the electrons $u_{e\parallel} \sim u_{i\parallel}$ by quasineutrality so
$u_{e\parallel} \sim v_{Ti} \ll v_{Te}$, which means we can neglect
$u_{e\parallel}$ in most of the terms for evolution of $f_e$.

Similar to the [ion shape function equation](@ref
ion_full_moment_kinetic_equation), but $\partial F_e/\partial t$ is negligible,
i.e.  electrons move on timescales faster than the system evolution.
```math
\begin{align}
\dot{z}_e \frac{\partial F_e}{\partial z}
+ \dot{w}_{\parallel,e} \frac{\partial F_e}{\partial w_\parallel}
+ \dot{w}_{\perp,e} \frac{\partial F_e}{\partial w_\perp}
&= \dot{F}_e + \mathcal{C}_{ee} + \mathcal{C}_{ei} + \mathcal{C}_{en} + \mathcal{S}_e
\end{align}
```
where
```math
\begin{align}
\dot{z}_e &= v_{Te} w_\parallel \\

\dot{w}_{\parallel,e} &= \frac{1}{n_e m_e v_{Te}} \frac{\partial p_{e\parallel}}{\partial z}
                         + \frac{w_\parallel}{3 p_e} \frac{\partial q_{e\parallel}}{\partial z}
                         - w_\parallel^2 \frac{\partial v_{T_e}}{\partial z} \nonumber \\
                  &\quad - \frac{1}{m_e n_e v_{Te}} (S_{e,\mathrm{mom}} - m_e u_{e\parallel} S_{e,n})
                         - \frac{w_\parallel}{2 p_e} (S_{e,p} - T_e S_{e,n}) \\

\dot{w}_{\perp,e} &= \frac{w_\perp}{3 p_e} \frac{\partial q_{e\parallel}}{\partial z}
                     - w_\perp w_\parallel \frac{\partial v_{Te}}{\partial z} \nonumber \\
              &\quad - \frac{w_\perp}{2 p_e} (S_{e,p} - T_e S_{e,n}) \\

\frac{\dot{F}_e}{F_e} &= w_\parallel \left( 3 \frac{\partial v_{Te}}{\partial z} - \frac{v_{Te}}{n_e} \frac{\partial n_e}{\partial z} \right)
                         - \frac{1}{p_e} \frac{\partial q_{e\parallel}}{\partial z} \nonumber \\
                  &\quad + \frac{3}{2 p_e} S_{e,p} - \frac{5}{2 n_e} S_{e,n} \\

\mathcal{C}_{ee} &= \frac{v_{Te}^3}{n_e} C_{K,ee} \\
                 &= -\frac{v_{Te}^3}{n_e} \nu_{ee} \left( f_e - \frac{n_e}{\pi^{3/2} v_{Te}^3} \exp\left( -\frac{|\boldsymbol{v} - u_{e\parallel}\hat{\boldsymbol{z}}|^2}{v_{Te}^2} \right) \right) \\
                 &= - \nu_{ee} \left( F_e - \frac{1}{\pi^{3/2}} \exp\left( -w^2 \right) \right) \\

\mathcal{C}_{ei} &= \frac{v_{Te}^3}{n_e} C_{K,ei} \\
                 &= -\frac{v_{Te}^3}{n_e} \nu_{ei} \left( f_e - \frac{n_e}{\pi^{3/2} v_{Te}^3} \exp\left( -\frac{|\boldsymbol{v} - u_{i\parallel}\hat{\boldsymbol{z}}|^2}{v_{Te}^2} \right) \right) \\
                 &= - \nu_{ei} \left( F_e - \frac{1}{\pi^{3/2}} \exp\left( -\left( w_\parallel - \frac{(u_{i\parallel} - u_{e\parallel})}{v_{Te}} \right)^2 - w_\perp^2 \right) \right) \\

\mathcal{C}_{en} &= \text{not implemented yet} \\

\mathcal{S}_{e} &= \frac{v_{Te}^3}{n_e} S_e \\
\end{align}
```
Although $S_e$ should mostly be small in $\sqrt{m_e/m_i}$, we keep it for the
case when the source is a Maxwellian with a temperature significantly higher
than the electron temperature because then the source can contribute
significantly to the high energy tail of electrons. As we keep $\mathcal S_e$,
to ensure that the first 3 moments of the shape function equation vanish, we
must also keep the source contributions to $\dot w_{\parallel,e}$, $\dot
w_{\perp,e}$, and $\dot F_e/F_e$ in the same form as the ion equations.

Reduction to 1D1V
-----------------

We can take the $T_{e\perp}=0$ limit of the equations and marginalise over
$v_\perp$/$w_\perp$ to reduce to 1D1V in a very similar way [as for the
ions](@ref ion_reduction_to_2d1v).

Quasineutrality and force balance keep the same form, and the energy equation
becomes
```math
\begin{align}
&\frac{3}{2} \frac{\partial p_e}{\partial t}
+ \frac{\partial q_{e\parallel}}{\partial z}
+ p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
+ \frac{3}{2} u_{e\parallel} \frac{\partial p_e}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + \int \frac{1}{2} m_e (v_\parallel - u_{e\parallel})^2 \bar{C}_{ei} dv_\parallel
        + \int \frac{1}{2} m_e (v_\parallel - u_{e\parallel})^2 \bar{C}_{en} dv_\parallel \nonumber \\
&\qquad + \frac{3}{2} S_{e,p} \\

&\frac{3}{2} \frac{\partial p_e}{\partial t}
+ \frac{\partial q_{e\parallel}}{\partial z}
+ p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
+ \frac{3}{2} u_{e\parallel} \frac{\partial p_e}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + 3 n_e \frac{m_e}{m_i} \nu_{ei} \left( T_i - T_e \right)
        + m_e n_e \nu_{ei} \left( u_{i\parallel} - u_{e_\parallel} \right)^2 \nonumber \\
&\qquad + \frac{3}{2} S_{e,p} \\
\end{align}
```

The kinetic equation for $\bar F_e = \int F_e d^2 w_\perp$ is
```math
\begin{align}
\dot{z}_e \frac{\partial \bar{F}_e}{\partial z}
+ \dot{w}_{\parallel,e} \frac{\partial \bar{F}_e}{\partial w_\parallel}
&= \dot{\bar{F}}_e
   + \bar{\mathcal{C}}_{ee} + \bar{\mathcal{C}}_{ei} + \bar{\mathcal{C}}_{en}
   + \bar{\mathcal{S}}_e
\end{align}
```
```math
\begin{align}
\dot{z}_e &= v_{Te} w_\parallel \\

\dot{w}_{\parallel,e} &= \frac{1}{n_e m_e v_{Te}} \frac{\partial p_{e\parallel}}{\partial z}
                         + \frac{w_\parallel}{3 p_e} \frac{\partial q_{e\parallel}}{\partial z}
                         - w_\parallel^2 \frac{\partial v_{T_e}}{\partial z} \nonumber \\
                  &\quad - \frac{1}{m_e n_e v_{Te}} (S_{e,\mathrm{mom}} - m_e u_{e\parallel} S_{e,n})
                         - \frac{w_\parallel}{2 p_e} (S_{i,p} - T_e S_{e,n}) \\

\frac{\dot{\bar{F}}_e}{\bar{F}_e} &= w_\parallel \left( \frac{\partial v_{Te}}{\partial z} - \frac{v_{Te}}{n_e} \frac{\partial n_e}{\partial z} \right)
                                     - \frac{1}{3 p_e} \frac{\partial q_{e\parallel}}{\partial z} \nonumber \\
                              &\quad + \frac{1}{2 p_e} S_{e,p}
                                     - \frac{3}{2 n_e} S_{e,n} \\

\bar{\mathcal{C}}_{ee} &= \frac{v_{Te}}{n_e} \bar{C}_{K,ee} \\
                       &= -\frac{v_{Te}}{n_e} \nu_{ee} \left( \bar{f}_e - \frac{n_e}{\sqrt{\pi} \sqrt{2 T_{e\parallel}/m_e}} \exp\left( -\frac{m_e (v_\parallel - u_{e\parallel})^2}{2 T_{e\parallel}} \right) \right) \\
                       &= - \nu_{ee} \left( \bar{F}_e - \frac{1}{\sqrt{3 \pi}} \exp\left( - \frac{w_\parallel^2}{3} \right) \right) \\

\bar{\mathcal{C}}_{ei} &= \frac{v_{Te}}{n_e} \bar{C}_{K,ei} \\
                       &= -\frac{v_{Te}}{n_e} \nu_{ei} \left( \bar{f}_e - \frac{n_e}{\sqrt{\pi} \sqrt{2 T_{e\parallel}/m_e}} \exp\left( -\frac{m_e (v_\parallel - u_{i\parallel})^2}{2 T_{e\parallel}} \right) \right) \\
                       &= - \nu_{ei} \left( \bar{F}_e - \frac{1}{\sqrt{3 \pi}} \exp\left( - \frac{\left( w_\parallel - \frac{(u_{i\parallel} - u_{e\parallel})}{v_{Te}} \right)^2}{3} \right) \right) \\

\bar{\mathcal{C}}_{en} &= \text{not implemented yet} \\

\bar{\mathcal{S}}_{e} &= \frac{v_{Te}}{n_e} \bar{S}_e = \frac{v_{Te}}{n_e} \int S_e d^2 v_\perp \\
\end{align}
```
recalling that $\int w_\perp \frac{\partial F_e}{\partial w_\perp} d^2 w_\perp
= 2 \bar F_e$ as for the ions, and noting that in the $T_{e,\perp} = 0$ limit,
$T_{e\parallel} = T_e/3$ so that $v_{Te}^2 = 2T_e/m_e = 6 T_{e\parallel}/m_e$.

Dimensionless equations
-----------------------

We make the equations dimensionless using the [conversions defined here](@ref
"Dimensionless equations for code").

### 1D2V

The moment equations become
```math
\begin{align}
\hat{n}_e &= \hat{n}_i \\

\hat{u}_{e\parallel} &= \hat{u}_{i\parallel} \\

\hat{E}_\parallel &= -\frac{1}{\hat{n}_e} \frac{\partial \hat{p}_{e\parallel}}{\partial \hat{z}}
                     + \frac{\hat{F}_{ei\parallel}}{\hat{n}_e}
                     + \frac{\hat{m}_e}{\hat{n}_e} \int \hat{v}_\parallel \hat{C}_{en} d^3 \hat{v}
                     + \frac{1}{\hat{n}_e} \hat{S}_{e,\mathrm{mom}} \\
\hat{E}_\parallel &= -\frac{1}{\hat{n}_e} \frac{\partial \hat{p}_{e\parallel}}{\partial \hat{z}}
                     + \hat{m}_e \hat{\nu}_{ei} \left( \hat{u}_{i\parallel} - \hat{u}_{e\parallel} \right)
                     + \frac{1}{\hat{n}_e} \hat{S}_{e,\mathrm{mom}} \\
\end{align}
```

```math
\begin{align}
&\frac{3}{2} \frac{\partial \hat{p}_e}{\partial \hat{t}}
+ \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
+ \hat{p}_{e\parallel} \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}}
+ \frac{3}{2} \hat{u}_{e\parallel} \frac{\partial \hat{p}_e}{\partial \hat{z}}
+ \frac{3}{2} \hat{p}_e \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{E}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{R}_\mathrm{ioniz}
        + \int \frac{1}{2} \hat{m}_e |\hat{\boldsymbol{v}} - \hat{u}_{e\parallel} \hat{\boldsymbol{z}}|^2 \hat{C}_{ei} d^3 v
        + \int \frac{1}{2} \hat{m}_e |\hat{\boldsymbol{v}} - \hat{u}_{e\parallel} \hat{\boldsymbol{z}}|^2 \hat{C}_{en} d^3 v \nonumber \\
&\qquad + \frac{3}{2} \hat{S}_{e,p} \\

&\frac{3}{2} \frac{\partial \hat{p}_e}{\partial \hat{t}}
+ \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
+ \hat{p}_{e\parallel} \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}}
+ \frac{3}{2} \hat{u}_{e\parallel} \frac{\partial \hat{p}_e}{\partial \hat{z}}
+ \frac{3}{2} \hat{p}_e \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{E}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{R}_\mathrm{ioniz}
        + 3 \hat{n}_e \frac{\hat{m}_e}{\hat{m}_i} \hat{\nu}_{ei} \left( \hat{T}_i - \hat{T}_e \right)
        + \hat{m}_e \hat{n}_e \hat{\nu}_{ei} \left( \hat{u}_{i\parallel} - \hat{u}_{e_\parallel} \right)^2 \nonumber \\
&\qquad + \frac{3}{2} \hat{S}_{e,p} \\
\end{align}
```

and the dimensionless kinetic equation is
```math
\begin{align}
\hat{\dot{z}}_e \frac{\partial F_e}{\partial \hat{z}}
+ \hat{\dot{w}}_{\parallel,e} \frac{\partial F_e}{\partial w_\parallel}
+ \hat{\dot{w}}_{\perp,e} \frac{\partial F_e}{\partial w_\perp}
&= \hat{\dot{F}}_e + \hat{\mathcal{C}}_{ee} + \hat{\mathcal{C}}_{ei} + \hat{\mathcal{C}}_{en} + \hat{\mathcal{S}}_e
\end{align}
```
where
```math
\begin{align}
\hat{\dot{z}}_e &= \hat{v}_{Te} w_\parallel \\

\hat{\dot{w}}_{\parallel,e} &= \frac{1}{\hat{n}_e \hat{m}_e \hat{v}_{Te}} \frac{\partial \hat{p}_{e\parallel}}{\partial \hat{z}}
                               + \frac{w_\parallel}{3 \hat{p}_e} \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
                               - w_\parallel^2 \frac{\partial \hat{v}_{T_e}}{\partial \hat{z}} \nonumber \\
                        &\quad - \frac{1}{\hat{m}_e \hat{n}_e \hat{v}_{Te}} (S_{e,\mathrm{mom}} - m_e u_{e\parallel} S_{e,n})
                               - \frac{w_\parallel}{2 \hat{p}_e} (\hat{S}_{e,p} - T_e S_{e,n}) \\

\hat{\dot{w}}_{\perp,e} &= \frac{w_\perp}{3 \hat{p}_e} \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
                           - w_\perp w_\parallel \frac{\partial \hat{v}_{Te}}{\partial \hat{z}} \nonumber \\
                    &\quad - \frac{w_\perp}{2 \hat{p}_e} ()\hat{S}_{e,p} - T_e \hat{S}_{e,n}) \\

\frac{\hat{\dot{F}}_e}{F_e} &= w_\parallel \left( 3 \frac{\partial \hat{v}_{Te}}{\partial \hat{z}} - \frac{\hat{v}_{Te}}{\hat{n}_e} \frac{\partial \hat{n}_e}{\partial \hat{z}} \right)
                               - \frac{1}{\hat{p}_e} \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}} \nonumber \\
                        &\quad + \frac{3}{2 \hat{p}_e} \hat{S}_{e,p}
                               - \frac{5}{2 \hat{n}_e} \hat{S}_{e,n} \\

\hat{\mathcal{C}}_{ee} &= \frac{\hat{v}_{Te}^3}{\hat{n}_e} \hat{C}_{K,ee} \\
                       &= -\frac{\hat{v}_{Te}^3}{\hat{n}_e} \hat{\nu}_{ee} \left( \hat{f}_e - \frac{\hat{n}_e}{\pi^{3/2} \hat{v}_{Te}^3} \exp\left( -\frac{|\hat{\boldsymbol{v}} - \hat{u}_{e\parallel}\hat{\boldsymbol{z}}|^2}{\hat{v}_{Te}^2} \right) \right) \\
                       &= - \hat{\nu}_{ee} \left( F_e - \frac{1}{\pi^{3/2}} \exp\left( -w^2 \right) \right) \\

\hat{\mathcal{C}}_{ei} &= \frac{\hat{v}_{Te}^3}{\hat{n}_e} \hat{C}_{K,ei} \\
                       &= -\frac{\hat{v}_{Te}^3}{\hat{n}_e} \hat{\nu}_{ei} \left( \hat{f}_e - \frac{\hat{n}_e}{\pi^{3/2} \hat{v}_{Te}^3} \exp\left( -\frac{|\hat{\boldsymbol{v}} - \hat{u}_{i\parallel}\hat{\boldsymbol{z}}|^2}{\hat{v}_{Te}^2} \right) \right) \\
                       &= - \hat{\nu}_{ei} \left( F_e - \frac{1}{\pi^{3/2}} \exp\left( -\left( w_\parallel - \frac{(\hat{u}_{i\parallel} - \hat{u}_{e\parallel})}{\hat{v}_{Te}} \right)^2 - w_\perp^2 \right) \right) \\

\hat{\mathcal{C}}_{en} &= \text{not implemented yet} \\

\hat{\mathcal{S}}_{e} &= \frac{\hat{v}_{Te}^3}{\hat{n}_e} \hat{S}_e \\
\end{align}
```

### 1D1V

In 1D1V the dimsionless energy equation is
```math
\begin{align}
&\frac{3}{2} \frac{\partial \hat{p}_e}{\partial \hat{t}}
+ \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
+ \hat{p}_{e\parallel} \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}}
+ \frac{3}{2} \hat{u}_{e\parallel} \frac{\partial \hat{p}_e}{\partial \hat{z}}
+ \frac{3}{2} \hat{p}_e \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{E}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{R}_\mathrm{ioniz}
        + \int \frac{1}{2} \hat{m}_e (\hat{v}_\parallel - \hat{u}_{e\parallel})^2 \hat{\bar{C}}_{ei} d\hat{v}_\parallel
        + \int \frac{1}{2} \hat{m}_e (\hat{v}_\parallel - \hat{u}_{e\parallel})^2 \hat{\bar{C}}_{en} d\hat{v}_\parallel \nonumber \\
&\qquad + \frac{3}{2} \hat{\bar{S}}_{e,p} \\

&\frac{3}{2} \frac{\partial \hat{p}_e}{\partial \hat{t}}
+ \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
+ \hat{p}_{e\parallel} \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}}
+ \frac{3}{2} \hat{u}_{e\parallel} \frac{\partial \hat{p}_e}{\partial \hat{z}}
+ \frac{3}{2} \hat{p}_e \frac{\partial \hat{u}_{e\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{E}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{R}_\mathrm{ioniz}
        + 3 \hat{n}_e \frac{\hat{m}_e}{\hat{m}_i} \hat{\nu}_{ei} \left( \hat{T}_i - \hat{T}_e \right)
        + \hat{m}_e \hat{n}_e \hat{\nu}_{ei} \left( \hat{u}_{i\parallel} - \hat{u}_{e_\parallel} \right)^2 \nonumber \\
&\qquad + \frac{3}{2} \hat{\bar{S}}_{e,p} \\
\end{align}
```

and the dimensionless kinetic equation is
```math
\begin{align}
\hat{\dot{z}}_e \frac{\partial \bar{F}_e}{\partial \hat{z}}
+ \hat{\dot{w}}_{\parallel,e} \frac{\partial \bar{F}_e}{\partial w_\parallel}
&= \hat{\dot{\bar{F}}}_e
   + \hat{\bar{\mathcal{C}}}_{ee} + \hat{\bar{\mathcal{C}}}_{ei} + \hat{\bar{\mathcal{C}}}_{en}
   + \hat{\bar{\mathcal{S}}}_e
\end{align}
```
```math
\begin{align}
\hat{\dot{z}}_e &= \hat{v}_{Te} w_\parallel \\

\hat{\dot{w}}_{\parallel,e} &= \frac{1}{\hat{n}_e \hat{m}_e \hat{v}_{Te}} \frac{\partial \hat{p}_{e\parallel}}{\partial \hat{z}}
                               + \frac{w_\parallel}{3 \hat{p}_e} \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}}
                               - w_\parallel^2 \frac{\partial \hat{v}_{T_e}}{\partial \hat{z}} \nonumber \\
                        &\quad - \frac{1}{\hat{m}_e \hat{n}_e \hat{v}_{Te}} (\hat{S}_{e,\mathrm{mom}} - \hat{m}_e \hat{u}_{e\parallel} \hat{S}_{e,n})
                               - \frac{w_\parallel}{2 \hat{p}_e} (\hat{S}_{e,p} - \hat{T}_e \hat{S}_{e,n}) \\

\frac{\hat{\dot{\bar{F}}}_e}{\bar{F}_e} &= w_\parallel \left( \frac{\partial \hat{v}_{Te}}{\partial \hat{z}} - \frac{\hat{v}_{Te}}{\hat{n}_e} \frac{\partial \hat{n}_e}{\partial \hat{z}} \right)
                                     - \frac{1}{3 \hat{p}_e} \frac{\partial \hat{q}_{e\parallel}}{\partial \hat{z}} \nonumber \\
                              &\quad + \frac{1}{2 \hat{p}_e} \hat{S}_{e,p}
                                     - \frac{3}{2 \hat{n}_e} \hat{S}_{e,n} \\

\hat{\bar{\mathcal{C}}}_{ee} &= \frac{\hat{v}_{Te}}{\hat{n}_e} \hat{\bar{C}}_{K,ee} \\
                             &= - \hat{\nu}_{ee} \left( \bar{F}_e - \frac{1}{\sqrt{3 \pi}} \exp\left( - \frac{w_\parallel^2}{3} \right) \right) \\

\hat{\bar{\mathcal{C}}}_{ei} &= \frac{\hat{v}_{Te}}{\hat{n}_e} \hat{\bar{C}}_{K,ei} \\
                             &= - \hat{\nu}_{ei} \left( \bar{F}_e - \frac{1}{\sqrt{3 \pi}} \exp\left( - \frac{\left( w_\parallel - \frac{(\hat{u}_{i\parallel} - \hat{u}_{e\parallel})}{\hat{v}_{Te}} \right)^2}{3} \right) \right) \\

\hat{\bar{\mathcal{C}}}_{en} &= \text{not implemented yet} \\

\hat{\bar{\mathcal{S}}}_{e} &= \frac{\hat{v}_{Te}}{\hat{n}_e} \hat{\bar{S}}_e = \frac{\hat{v}_{Te}}{\hat{n}_e} \int \hat{S}_e d^2 \hat{v}_\perp \\
\end{align}
```

The conversion to the dimensionless equations in the 1D1V Excalibur reports,
and the original version of the code, uses the [conversions given here](@ref
"Conversion to old dimensionless equations").

Old 1D1V kinetic electron equations
-----------------------------------

These were the form of equations implemented in the code for kinetic electrons
before PR #322, April 2025.

```@raw html
<details>
<summary style="text-align:center">[ notes using old definitions and dimensionless variables ]</summary>
```
```math
\begin{align}
n_e &= n_i \\
\end{align}
```

```math
\begin{align}
\Gamma_{\parallel,\mathrm{net}}(z=-L_z/2) &= (n_i(z=-L_z/2) u_{i\parallel}(z=-L_z/2) - n_e(z=-L_z/2) u_{e\parallel}(z=-L_z/2)) = 0 \\
u_{e\parallel} &= \frac{\left( -\Gamma_{\parallel,\mathrm{net}}(z=-L_z/2) + n_i u_i \right)}{n_e} \\
\end{align}
```

```math
\begin{align}
E_\parallel &= - \frac{2}{n_e} \frac{\partial p_{e\parallel}}{\partial z} \\
\end{align}
```

```math
\begin{align}
\frac{\partial p_{e\parallel}}{\partial t}
  &= -u_{e\parallel} \frac{\partial p_{e\parallel}}{\partial z}
     - 3 p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
     - \frac{\partial q_{e\parallel}}{\partial z}
     + D_{p_e,z} \frac{\partial^2 p_{e\parallel}}{\partial z^2} + S_{p,e} \\
\end{align}
```
where $D_{p_e,z}$ is a numerical diffusion coefficient, which we usually leave as 0.

```math
\begin{align}
\frac{\partial g_e}{\partial t}
  + \dot{z} \frac{\partial g_e}{\partial z}
  + \dot{w}_\parallel \frac{\partial g_e}{\partial w_\parallel}
&= \dot{g} + \mathcal{D}_\mathrm{num} + \mathcal{C}_{K,e} + \mathcal{S}_e
\end{align}
```
where
```math
\begin{align}
\dot{z} &= v_{Te} w_\parallel + u_{e\parallel} \\
\end{align}
```
```math
\begin{align}
\dot{w}_\parallel
&= \frac{v_{Te}}{2 p_{e\parallel}} \frac{\partial p_{e\parallel}}{\partial z} + \frac{w_\parallel}{2 p_{e\parallel}} \frac{\partial q_{e\parallel}}{\partial z}
   - w_\parallel^2 \frac{\partial v_{Te}}{\partial z}
   + \frac{S_{n,e} u_{e\parallel}}{n_e v_{Te}}
   - w_\parallel \frac{S_{p,e} + 2 u_{e\parallel} S_{\mathrm{mom},e}}{2 p_{e\parallel}}
   + w_\parallel \frac{S_n}{2 n_e}
\end{align}
```
Think this is missing a $S_{\mathrm{mom},e}$ term that is not multiplied by
$w_\parallel$ and has an extra one that is multiplied by $w_\parallel$, but so
far $S_{\mathrm{mom},s}$ is always zero anyway, so this doesn't matter.

```math
\begin{align}
\frac{\dot{g}_e}{g_e}
&= -\frac{1}{2 p_{e\parallel}} \frac{\partial q_{e\parallel}}{\partial z}
   - w_\parallel v_{Te} \left( \frac{1}{n_e} \frac{\partial n_e}{\partial z}
                               - \frac{1}{v_{Te}} \frac{\partial v_{Te}}{\partial z} \right)
   - \frac{3 S_n}{2 n_e} + \frac{S_{p,e}/2 + S_{\mathrm{mom},e}}{p_{e\parallel}}
\end{align}
```
```math
\begin{align}
\mathcal{D}_\mathrm{num}
&= D_{w_\parallel,e} \frac{\partial^2 g_e}{\partial w_\parallel^2}
\end{align}
```
```math
\begin{align}
\mathcal{C}_{K,e}
&= \nu_{ee} \left[ g_e - \exp\left( -w_\parallel^2 \right) \right]
   + \nu_{ei} \left[ g_e - \exp\left( -\left( w_\parallel + (u_{i\parallel} - u_{e\parallel})/v_{Te} \right)^2 \right) \right]
\end{align}
```
```math
\begin{align}
\mathcal{S}_e
&= \frac{v_{Te}}{n_e} S_e
\end{align}
```
```@raw html
</details>
```
