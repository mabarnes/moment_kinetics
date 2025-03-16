Moment kinetic equations for electrons
======================================

For electrons, we can convert the equations for ions ([Ion moment
equations](@ref) and [Full moment-kinetics (separate $n_i$, $u_{i\parallel}$
and $p_i$)](@ref ion_full_moment_kinetic_equation)) by taking the charge to
have the opposite sign (which only appears in the parallel momentum equation
for the 1D2V system), changing the mass to $m_e$, and swapping ion-$s$
collision operators for electron-$s$ collision operators. We then use
quasineutrality and a mass ratio expansion to simplify the equations.

Mass ratio expansion
--------------------

$m_e/m_i$ is a small parameter, so $v_{Ti}/v_{Te} \sim \sqrt{m_e/m_i}$ is also
a small parameter, assuming the temperatures are comparable $T_e \sim T_i$.

We assume that sources and any turbulence act on ion timescales, and we are not
interested in any initial transient that might happen on electron timescales,
so the fastest possible time variation is
```math
\begin{align}
\frac{\partial}{\partial t} &\sim \frac{v_{Ti}}{L_\parallel} \sim \sqrt{\frac{m_e}{m_i}} \frac{v_{Te}}{L_\parallel} \\
\end{align}
```

We will see from the [Continuity equation](@ref) that $u_e \sim u_i \sim
v_{Ti}$, so $u_e/v_{Te} \sim \sqrt{m_e/m_i}$.

The sources of particles and energy for the electrons are assumed comparable in
size to those for the ions, and must balance a loss rate to the walls whose
magnitude is set by the ion thermal speed, so for both electrons and ions
```math
\begin{align}
\int S_s d^3 v \sim \frac{n_s v_{Ti}}{L_\parallel} \\
\int m_e v_\parallel S_s d^3 v \lesssim \frac{n_s v_{Ti}^2}{L_\parallel} \\
\int \frac{1}{2} m_s v^2 S_s d^3 v \sim \frac{n_s T_s v_{Ti}}{L_\parallel} \\
S_s \sim \frac{n_s v_{Ti} F_s}{L_\parallel} \\
\end{align}
```
The momentum source on the second line is likely to be exactly zero for a
plausible source (in which case its size would be $\sim 0$), but at maximum
would be comparable to injecting particles at the ion thermal speed (which
would give a size $\sim n_s v_{Ti}^2 / L_\parallel$) - this difference is why
we put '$\lesssim$'.

Continuity equation
-------------------

```math
\begin{align}
\frac{\partial n_e}{\partial t} + \frac{\partial}{\partial z}\left( n_e u_{e\parallel} \right)
    = R_\mathrm{ioniz} n_e n_n + \int S_e d^3 v
\end{align}
```
Using quasineutrality $n_e = n_i$, so $\partial n_e/\partial t = \partial
n_i/\partial t$ and that the source terms do not add any charge, so $\int S_e
d^3 v = \int S_i d^3 v$, we can subtract the electron continuity equation from
the ion one to find
```math
\begin{align}
\frac{\partial}{\partial z}\left( n_i u_{i\parallel} - n_e u_{e\parallel} \right) = 0
\end{align}
```
which we could write as $\nabla\cdot\boldsymbol J_\parallel = 0$, and tells us
that the parallel current is constant along field lines, and so is determined
(in 1D) only by the sheath boundary conditions. For the moment, we take this to
mean that $J_\parallel = 0$ so $u_e = u_i$ as $n_e = n_i$. Perpendicular
currents due to drifts or polarization effects (etc?) in 2D/3D would alter
this, making the parallel current not constant. Potentially, temperature
differences even in 1D could give a non-zero parallel current, but this is not
yet accounted for (so far we only deal with simulations that are symmetric
about the midpoint in $z$ and so have the same temperature at both targets).

Parallel momentum equation
--------------------------

```math
\begin{align}
& \underbrace{m_e \frac{\partial}{\partial t}(n_e u_{e\parallel})}_{\sim \frac{m_e n_e v_{Ti}^2}{L_\parallel}}
  + \underbrace{m_e \frac{\partial}{\partial z}(n_e u_{e\parallel}^2)}_{\sim \frac{m_e n_e v_{Ti}^2}{L_\parallel}}
  + \underbrace{\frac{\partial p_{e\parallel}}{\partial z}}_{\sim \frac{n_e T_e}{L_\parallel} \sim \frac{m_e n_e v_{Te}^2}{L_\parallel} \sim \frac{m_i n_e v_{Ti}^2}{L_\parallel}}
  - \underbrace{e n_e \frac{\partial \phi}{\partial z}}_{\sim \frac{n_e T_e}{L_\parallel} \sim \frac{m_e n_e v_{Te}^2}{L_\parallel} \sim \frac{m_i n_e v_{Ti}^2}{L_\parallel}} \nonumber \\
&\quad= \underbrace{m_e \int v_\parallel C_{ei}(f_e,f_i) d^3 v}_{\sim m_e \nu_{ei} n_e v_{Ti}}
        + \underbrace{m_e \int v_\parallel C_{en}(f_e,f_n) d^3 v}_{\sim m_e \nu_{en} n_e v_{Ti}}
        + \underbrace{m_e \int v_\parallel S_e d^3 v}_{\lesssim \frac{m_e n_e v_{Ti}^2}{L_\parallel} } \\
\end{align}
```
where $C_{en}(f_e,f_n)$ represents any electron-neutral interactions (elastic
scattering, ionization, excitation).

Assume for now that $\{\nu_{ei},\nu_{en}\} \sim v_{Ti}/L_\parallel$ (this might
need to be modified eventually, for example for a very strongly collisional
plasma, or a deeply detached plasma with very strong neutral interactions).
Then
```math
\begin{align}
\frac{\partial p_{e\parallel}}{\partial z} - e n_e \frac{\partial \phi}{\partial z} + O\left( \frac{m_e}{m_i} \right) = 0
\end{align}
```

Energy equation
---------------

```math
\begin{align}
& \underbrace{\frac{3}{2} \frac{\partial p_e}{\partial t}}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}}
  + \underbrace{\frac{\partial q_{e\parallel}}{\partial z}}_{\sim \frac{n_e T_e v_{Te}}{L_\parallel}}
  + \underbrace{p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}}
  + \underbrace{\frac{3}{2} u_{e\parallel} \frac{\partial p_e}{\partial z}}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}}
  + \underbrace{\frac{3}{2} p_e \frac{\partial u_{e\parallel}}{\partial z}}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}} \nonumber \\
&\quad= - \underbrace{m_e \int (\boldsymbol{v} - \boldsymbol{u}_{e\parallel})^2 C_{ei}(f_e,f_i) d^3 v}_{\sim \nu_{ei} n_e T_e}
        + \underbrace{m_e \int (\boldsymbol{v} - \boldsymbol{u}_{e\parallel})^2 C_{en}(f_e,f_n) d^3 v}_{\sim \nu_{en} n_e T_e} \nonumber \\
&\qquad+ \underbrace{\frac{1}{2} \int m_e v^2 S_e d^3 v}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}}
       - \underbrace{m_e u_{e\parallel} \int v_\parallel S_e d^3 v}_{\sim \frac{m_e n_e T_e v_{Ti}}{L_\parallel}}
       + \underbrace{\frac{1}{2} m_e u_{e\parallel}^2 \int S_e d^3 v}_{\sim \frac{n_e T_e v_{Ti}}{L_\parallel}} \\
\end{align}
```

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
  &= u_{e\parallel} \frac{\partial p_{e\parallel}}{\partial z}
     + 3 p_{e\parallel} \frac{\partial u_{e\parallel}}{\partial z}
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
&= \nu_{ee} (g_e - \exp\left( -w_\parallel^2 \right)
   + \nu_{ei} (g_e - \exp\left( -\left( w_\parallel + (u_{i\parallel} - u_{e\parallel})/v_{Te} \right)^2 \right)
\end{align}
```
[`moment_kinetics.external_sources.external_electron_source!`](@ref)
```math
\begin{align}
\mathcal{S}_e
&= \frac{v_{Te}}{n_e} S_e
\end{align}
```
