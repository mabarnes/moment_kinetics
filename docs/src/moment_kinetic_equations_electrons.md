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
&= n_e n_n R_\mathrm{ioniz} + \int S_e d^3 v
\end{align}
```
Subtracting from ion continuity and using quasineutrality
```math
\begin{align}
\frac{\partial}{\partial z}\left(n_e (u_{i\parallel} - u_{e\parallel}) \right)
&= \int S_i d^3 v - \int S_e d^3 v
\end{align}
```
giving an equation for $u_{e\parallel}$. Assuming that the sources do not add
any charge $\int S_i d^3 v = \int S_e d^3 v$ then implies that
$u_{e\parallel} = u_{i\parallel} + u_0$ assuming in addition (at least for the
moment) that there is no current through the sheath sets $u_0 = 0$ so that
```math
\begin{align}
u_{e\parallel} = u_{i\parallel}
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
     + m_e \int v_\parallel S_e d^3 v
\end{align}
```
where $C_{en}$ represents elastic electron-neutral collisions (for collisions
see [Electron collisions](@ref)), which gives an equation for
$E_\parallel = -\partial \phi/\partial z$
```math
\begin{align}
e E_\parallel &= -\frac{\partial p_{e\parallel}}{\partial z}
                 + F_{ei\parallel}
                 + m_e \int v_\parallel C_{en} d^3 v
                 + m_e \int v_\parallel S_e d^3 v
e E_\parallel &= -\frac{\partial p_{e\parallel}}{\partial z}
                 + m_e n_e \nu_{ei} \left( u_{i\parallel} - u_{e\parallel} \right)
                 + m_e \int v_\parallel S_e d^3 v
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
+ \frac{3}{2} u_{e\parallel} \frac{\partial{p_e}}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + \int \frac{1}{2} m_e |\boldsymbol{v} - u_{e\parallel} \hat{\boldsymbol{z}}|^2 C_{ei} d^3 v
        + \int \frac{1}{2} m_e |\boldsymbol{v} - u_{e\parallel} \hat{\boldsymbol{z}}|^2 C_{en} d^3 v \nonumber \\
&\qquad + \frac{1}{2} \int m_e v^2 S_e d^3 v
        - m_e u_{e\parallel} \int v_\parallel S_e d^3 v
        + \frac{1}{2} m_e u_{e\parallel}^2 \int S_e d^3 v \\

&\frac{3}{2} \frac{\partial p_e}{\partial t}
+ \frac{\partial q_{e\parallel}}{\partial z}
+ p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
+ \frac{3}{2} u_{e\parallel} \frac{\partial{p_e}}{\partial z}
+ \frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z} \nonumber \\
&\quad= -E_\mathrm{ioniz} n_e n_n R_\mathrm{ioniz}
        + 3 n_e \frac{m_e}{m_i} \nu_{ei} \left( T_i - T_e \right)
        + m_e n_e \nu_{ei} \left( u_{i\parallel} - u_{e_\parallel} \right)^2 \nonumber \\
&\qquad + \frac{1}{2} \int m_e v^2 S_e d^3 v
        - m_e u_{e\parallel} \int v_\parallel S_e d^3 v
        + \frac{1}{2} m_e u_{e\parallel}^2 \int S_e d^3 v \\
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
                         + \frac{2 w_\parallel}{3 n_e m_e v_Te^2} \frac{\partial q_{e\parallel}}{\partial z}
                         - w_\parallel^2 \frac{\partial v_{T_e}}{\partial z} \nonumber \\
                  &\quad - \frac{1}{n_e v_{Te}} \int v_\parallel S_e d^3 v
                         - \frac{w_\parallel}{6 p_e} \int m_e v^2 S_e d^3 v
                         + \frac{w_\parallel}{3 p_e} \frac{3}{2} T_e \int S_e d^3 v \\

\dot{w}_{\perp,e} &= \frac{2 w_\perp}{3 m_e n_e v_{Te}^2} \frac{\partial q_{e\parallel}}{\partial z}
                     - w_\perp w_\parallel \frac{\partial v_{Te}}{\partial z} \nonumber \\
              &\quad - \frac{w_\perp}{6 p_e} \int m_e v^2 S_e d^3 v
                     + \frac{w_\perp}{3 p_e} \frac{3}{2} T_e \int S_e d^3 v \\

\frac{\dot{F}_e}{F_e} &= w_\parallel \left( 3 \frac{\partial v_{Te}}{\partial z} - \frac{v_{Te}}{n_e} \frac{\partial n_e}{\partial z} \right)
                         - \frac{2}{m_e n_e v_{Te}^2} \frac{\partial q_{e\parallel}}{\partial z} \nonumber \\
                  &\quad + \frac{1}{2 p_e} \int m_e v^2 S_e d^3 v
                         - \frac{1}{p_e} \frac{3}{2} T_e \int S_e d^3 v
                         - \frac{1}{n_e} \int S_e d^3 v \\

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
w_{\perp,e}$, and $\dot F_e/F_e$ in the same form as the ion equations,
although we can drop the terms that would be multiplied by $u_{e\parallel}$.

Implemented, dimensionless, 1D1V equations
------------------------------------------

These are the equations as currently implemented in the code for kinetic electrons (16/3/2025).

[`moment_kinetics.electron_fluid_equations.calculate_electron_density!`](@ref)
```math
\begin{align}
n_e &= n_i \\
\end{align}
```

[`moment_kinetics.electron_fluid_equations.calculate_electron_upar_from_charge_conservation!`](@ref)
```math
\begin{align}
\Gamma_{\parallel,\mathrm{net}}(z=-L_z/2) &= (n_i(z=-L_z/2) u_{i\parallel}(z=-L_z/2) - n_e(z=-L_z/2) u_{e\parallel}(z=-L_z/2)) = 0 \\
u_{e\parallel} &= \frac{\left( -\Gamma_{\parallel,\mathrm{net}}(z=-L_z/2) + n_i u_i \right)}{n_e} \\
\end{align}
```

[`moment_kinetics.electron_fluid_equations.calculate_Epar_from_electron_force_balance!`](@ref)
```math
\begin{align}
E_\parallel &= - \frac{2}{n_e} \frac{\partial p_{e\parallel}}{\partial z} \\
\end{align}
```

[`moment_kinetics.electron_fluid_equations.electron_energy_equation_no_r!`](@ref)
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

[`moment_kinetics.electron_kinetic_equation.electron_kinetic_equation_euler_update!`](@ref)
```math
\begin{align}
\frac{\partial g_e}{\partial t}
  + \dot{z} \frac{\partial g_e}{\partial z}
  + \dot{w}_\parallel \frac{\partial g_e}{\partial w_\parallel}
&= \dot{g} + \mathcal{D}_\mathrm{num} + \mathcal{C}_{K,e} + \mathcal{S}_e
\end{align}
```
where
[`moment_kinetics.electron_z_advection.update_electron_speed_z!`](@ref)
```math
\begin{align}
\dot{z} &= v_{Te} w_\parallel + u_{e\parallel} \\
\end{align}
```
[`moment_kinetics.electron_vpa_advection.update_electron_speed_vpa!`](@ref)
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
[`moment_kinetics.electron_kinetic_equation.add_contribution_from_pdf_term!`](@ref)
```math
\begin{align}
\frac{\dot{g}_e}{g_e}
&= -\frac{1}{2 p_{e\parallel}} \frac{\partial q_{e\parallel}}{\partial z}
   - w_\parallel v_{Te} \left( \frac{1}{n_e} \frac{\partial n_e}{\partial z}
                               - \frac{1}{v_{Te}} \frac{\partial v_{Te}}{\partial z} \right)
   - \frac{3 S_n}{2 n_e} + \frac{S_{p,e}/2 + S_{\mathrm{mom},e}}{p_{e\parallel}}
\end{align}
```
[`moment_kinetics.electron_kinetic_equation.add_dissipation_term!`](@ref)
```math
\begin{align}
\mathcal{D}_\mathrm{num}
&= D_{w_\parallel,e} \frac{\partial^2 g_e}{\partial w_\parallel^2}
\end{align}
```
[`moment_kinetics.krook_collisions.electron_krook_collisions!`](@ref)
```math
\begin{align}
\mathcal{C}_{K,e}
&= \nu_{ee} \left[ g_e - \exp\left( -w_\parallel^2 \right) \right]
   + \nu_{ei} \left[ g_e - \exp\left( -\left( w_\parallel + (u_{i\parallel} - u_{e\parallel})/v_{Te} \right)^2 \right) \right]
\end{align}
```
[`moment_kinetics.external_sources.external_electron_source!`](@ref)
```math
\begin{align}
\mathcal{S}_e
&= \frac{v_{Te}}{n_e} S_e
\end{align}
```
