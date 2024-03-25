External sources
================

Sometimes it is useful to have a source term for the plasma or neutrals (the
$S_i$ and $S_n$ of [Moment kinetic equations](@ref)). The currently-implemented
source term has the form of a Maxwellian with constant temperature and
spatially-varying amplitude
```math
\begin{align}
S_i &= A_i(r,z) \frac{1}{(\pi)^{3/2} (2 T_{\mathrm{source},i} / m_i)^{3/2}} \exp\left( -\frac{(v_\perp^2 + v_\parallel^2)}{T_{\mathrm{source},i}} \right) \\
S_n &= A_n(r,z) \frac{1}{(\pi)^{3/2} (2 T_{\mathrm{source},n} / m_n)^{3/2}} \exp\left( -\frac{(v_\zeta^2 + v_r^2 + v_z^2)}{T_{\mathrm{source},n}} \right)
\end{align}
```
or in 1V simulations that do not include $v_\perp$, $v_\zeta$, $v_r$ dimensions
```math
\begin{align}
S_i &= A_i(r,z) \frac{1}{\sqrt{\pi} \sqrt{2 T_{\mathrm{source},i} / m_i}} \exp\left( -\frac{v_\perp^2}{T_{\mathrm{source},i}} \right) \\
S_n &= A_n(r,z) \frac{1}{\sqrt{\pi} \sqrt{2 T_{\mathrm{source},n} / m_n}} \exp\left( -\frac{v_z^2}{T_{\mathrm{source},n}} \right)
\end{align}
```

The sources are controlled by options in the `[ion_source]` and
`[neutral_source]` sections of the input file. The source terms are enabled by
setting `active = true`. The constant temperature is set with the `source_T`
option (default is 1 for ions and $T_\mathrm{wall}$ for neutrals). The
amplitude can be set or controlled in various ways depending on the
`source_type` setting, as explained in the following subsection.

Note that all the settings mentioned below have values given in normalised
units (in the same way as the settings for initial profiles, etc.).

Amplitude
---------

### Fixed amplitude (default)

When `source_type = "Maxwellian"` (the default), the amplitude of the source is
fixed in time and controled by the profile options. The profile has the form
```math
A(r,z) = A_0 R(r) Z(z)
```
where $A_0$ is given by the `source_strength` option. $R(r)$ and $Z(z)$ are
controlled by the `r_profile` and `z_profile` options respectively. The
available options for either are the same, so letting `x` stand for either of
`r` or `z`, and `X` for the corresponding `R` or `Z`:
* `x_profile = "constant"` (the default) means $X(x)=1$.
* `x_profile = "gaussian"` means
    $X(x) = (1 - X_\mathrm{min}) \exp\left( -\left(\frac{x}{w}\right)^2 \right) + X_\mathrm{min}$
    where $X_\mathrm{min}$ is set by `x_relative_minimum` and $w$ is set by
    `x_width`.
* `x_profile = "parabolic"` means
    $P(x) = \left( 1 - \left(\frac{2x}{w}\right)^2 \right)$, 
    $X(x) = (1 - X_\mathrm{min}) H(P(x)) P(x) + X_\mathrm{min}$
    where $X_\mathrm{min}$ is set by `x_relative_minimum` and $w$ is set by
    `x_width`. The effect of the step function $H$ is to let the profile be a
    quadratic in the range $-w/2 < x < w/2$, but equal to a floor (by default
    0, so that the source is just not allowed to become negative) outside that
    range.

### Midpoint density controller

When `source_type = "density_midpoint_control"` a PI controller
([Wikipedia](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller))
is used to control the ion/neutral density. The 'midpoint' for the purposes of
this controller is the point on the grid where $r=0$ and $z=0$ (there must be a
grid point there).

The spatial profile of the source ($R(r)$ and $Z(z)$) is set in the same way as
for the 'Fixed amplitude' source (see above), but now the prefactor changes
with time
```math
A(r,z) = A_0(t) R(r) Z(z).
```
The prefactor $A_0(t)$ is controlled to set the midpoint density to some value
$n(r=0,z=0)\rightarrow n_\mathrm{PI}$ where $n_\mathrm{PI}$ is set by
`PI_density_target_amplitude`. Specifically,
```math
\begin{align}
  A_0(t) &= \mathtt{max}\left(P(n_\mathrm{PI} - n(r=0,z=0)) + \iota(t), 0\right) \\
  \frac{\partial \iota}{\partial t} &= I(n_\mathrm{PI} - n(r=0,z=0)).
\end{align}
```
The 'proportional' coefficient $P$ is set by `PI_density_controller_P` and the
'integral' coefficient $I$ is set by `PI_density_controller_I`. The
$\mathrm{max}(\ldots,0)$ is to ensure that the 'source term' is never negative
(i.e. a sink), to avoid the possibility of driving the system towards negative
density.

### Density profile controller

When `source_type = "density_profile_control"` a PI controller
([Wikipedia](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller))
is used to control the ion/neutral density profile.

The target profile is
```math
n_\mathrm{PI}(r,z) = n_{\mathrm{PI},0} R(r) Z(z)
```
where $n_{\mathrm{PI},0}$ is set by `PI_density_target_amplitude` and $R(r)$
and $Z(z)$ are set as described in [Fixed amplitude (default)](@ref).

The source amplitude $A(r,z)$ is controlled to set the density profile to
$n(r,z)\rightarrow n_\mathrm{PI}(r,z)$. Specifically,
```math
\begin{align}
  A(r,z) &= \mathtt{max}\left(P(n_\mathrm{PI}(r,z) - n(r,z)) + \iota(t,r,z), 0\right) \\
  \frac{\partial \iota(t,r,z)}{\partial t} &= I(n_\mathrm{PI}(r,z) - n(r,z)).
\end{align}
```
The 'proportional' coefficient $P$ is set by `PI_density_controller_P` and the
'integral' coefficient $I$ is set by `PI_density_controller_I`. The
$\mathrm{max}(\ldots,0)$ is to ensure that the 'source term' is never negative
(i.e. a sink), to avoid the possibility of driving the system towards negative
density.

### Recycling

The source of neutrals can be set so that some fraction of the flux of ions to
the walls is recycled into the volume of the domain as neutrals by using the
`source_type = "recycling"` option.

The profile is set up whose spatial integral is 1
```math
A(r,z) = A_0 R(r) Z(z)
```
where $A_0 = \left[\int dr\,dz\, R(r) Z(z)\right]^{-1}$ and $R(r)$ and $Z(z)$
are set as described in [Fixed amplitude (default)](@ref). The source is
```math
S_n(t,r,z) = F(t) A(r,z)
```
where $F(t)$ is the sum of the integrated ion flux to the lower and upper
targets.

!!! warning
    The target flux calculated for this controller does not account for
    magnetic field lines that are not perpendicular to the wall, or for drifts
    to the target, so needs updating (within
    [`moment_kinetics.external_sources.external_neutral_source_controller!`](@ref))
    to be used in 2D simulations.

### Energy source

When `source_type = "energy"`, rather than just adding particles with
temperature $T_\mathrm{source},s$, the existing plasma or neutrals in the
domain are swapped with plasma/neutrals from a Maxwellian with
$T_\mathrm{source},s$, so that the density is unchanged, but energy is added
(or potentially removed if the plasma/neutrals are hotter than
$T_\mathrm{source},s$).
```math
\begin{align}
S_i &= A_i(r,z) \left[ \frac{1}{(\pi)^{3/2} (2 T_{\mathrm{source},i} / m_i)^{3/2}} \exp\left( -\frac{(v_\perp^2 + v_\parallel^2)}{T_{\mathrm{source},i}} - f_i(v_\perp, v_\parallel) \right) \right] \\
S_n &= A_n(r,z) \left[ \frac{1}{(\pi)^{3/2} (2 T_{\mathrm{source},n} / m_n)^{3/2}} \exp\left( -\frac{(v_\zeta^2 + v_r^2 + v_z^2)}{T_{\mathrm{source},n}} - f_n(v_\zeta, v_r, v_z) \right) \right]
\end{align}
```
or in 1V simulations
```math
\begin{align}
S_i &= A_i(r,z) \left[ \frac{1}{sqrt{\pi} \sqrt{2 T_{\mathrm{source},i} / m_i}} \exp\left( -\frac{v_\perp^2}{T_{\mathrm{source},i}} - f_i(v_\parallel) \right) \right] \\
S_n &= A_n(r,z) \left[ \frac{1}{sqrt{\pi} \sqrt{2 T_{\mathrm{source},n} / m_n}} \exp\left( -\frac{v_z^2}{T_{\mathrm{source},n}} - f_n(v_z) \right) \right]
\end{align}
```
Note that this source does not give a fixed power input (although that might be
a nice feature to have), it just swaps plasma/neutral particles at a constant
rate.

API
---

See [external\_sources](@ref).
