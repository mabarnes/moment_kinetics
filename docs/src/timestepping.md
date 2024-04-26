Timestepping
============

Timestepping in `moment_kinetics` can be done with various explicit Runge-Kutta
(RK) schemes. The default is a fixed-timestep 3rd-order, 4-stage, strong
stability preserving (SSP) RK scheme.

Several schemes (including all the fixed-timestep schemes) use a 'low storage'
option, where only values from the first stage and previous stage are required
for each stage update[^1].

[^1]: At present, we take advantage of this property to reduce the
    number of computations in the RK update step, but do not actually reduce the
    memory usage - we still store the results from every RK stage. It would be
    fairly straightforward to save memory, but would only reduce from 4 copies to 3
    for the standard cases, so not a big saving.

Fixed-timestep schemes
----------------------

The fixed timestep schemes use a constant `dt`, specified in the input file,
for the whole simulation. The available types are:
* "SSPRK1" - forward Euler method
* "SSPRK2" - Heun's method
* "SSPRK3" - a 3-stage, 3rd order method, see [this Wikipedia
  list](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
* "SSPRK4" - the default, a 4-stage, 3rd order method, see [R.J. Spiteri, and
  S.J.  Ruuth. "A new class of optimal high-order strong-stability-preserving
  time discretization methods." SIAM Journal on Numerical Analysis 40.2 (2002):
  469-491., referenced in Dale E. Durran, “Numerical Methods for Fluid
  Dynamics”, Springer. Second Edition].

Adaptive-timestep schemes
-------------------------

Several SSP schemes are included from [Fekete, Conde and Shadid, "Embedded
pairs for optimal explicit strong stability preserving Runge-Kutta methods",
Journal of Computational and Applied Mathematics 421 (2022) 114325,
<https://doi.org/10.1016/j.cam.2022.114325>]:
* "Fekete4(3)" a 4-stage, 3rd order method, the recommended 3rd order method in
  Fekete et al. Identical to the default "SSPRK4" fixed-step method, but with
  an embedded 2nd order method used to provide error control for adaptive
  timestepping. This is probably a good first choice for an adaptive timestep
  method.
* "Fekete4(2)" a 4-stage, 2nd order method, the recommended 2nd order method in
  Fekete et al.
* "Fekete10(4)" a 10-stage, 4th order method, the recommended 4th order method
  in Fekete et al. May allow longer timesteps than "Fekete4(3)", but probably
  not any faster as more stages are required per timestep. However, if very
  high accuracy is required (very tight `rtol` and `atol` tolerances), the
  higher accuracy may be an advantage.
* "Fekete6(4)" a 6-stage, 4th order method.

The classic "Runge-Kutta-Fehlberg" method
[<https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method>,
'COEFFICIENTS FOR RK4(5), FORMULA 2 Table III in Fehlberg' - note the Wikipedia
page seems to have a typo in one of the error coefficients, see comment in
`utils/calculate_rk_coeffs.jl`] is also provided as "RKF5(4)". This method
seems to require a significantly smaller timestep to be stable than the SSP
methods from Fekete et al., but might be useful if very high accuracy is
required as it is a 5th-order accurate method. It uses 6 stages per step.

### Algorithm for choosing the next timestep

These adaptive timestepping methods use several criteria to set or limit the
timestep:
* Truncation error, which is estimated by the difference between the higher and
  lower order methods of an 'embedded pair'. The timestep size needed to
  maintain a specified accuracy can be estimated from the size of the trucation
  error (knowing the order of accuracy of the method), as described for example
  in Fehlberg et al. This estimate is used unless it is larger than any of the
  following limits. The error limit is set by relative tolerance "rtol" and
  absolute tolerance "atol" parameters. For each variable $X$ the error metric
  (calculated in [`moment_kinetics.time_advance.local_error_norm`](@ref) is the
  root-mean-square (RMS, or 'L2 norm') of $\epsilon$:
  ```math
  \epsilon = \frac{E_{X}}{(\mathtt{rtol}*|X| + \mathtt{atol})}
  ```
  where $E_{X}$ is the truncation error estimate for $X$. If the RMS of
  $\epsilon$, averaged over all evolved variables, is greater than 1, then the
  step is considered 'failed' and is re-done with a shorter timestep (set by
  the lower of half of the failed timestep, or the timestep calculated using
  the estimate based on $\epsilon$).
* CFL criteria
  [<https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition>].
  These are estimated for the spatial advection and velocity-space advection
  terms in the kinetic equation(s), using the methods
  [`moment_kinetics.utils.get_minimum_CFL_z`](@ref),
  [`moment_kinetics.utils.get_minimum_CFL_vpa`](@ref),
  [`moment_kinetics.utils.get_minimum_CFL_neutral_z`](@ref),
  [`moment_kinetics.utils.get_minimum_CFL_neutral_vz`](@ref). These estimates
  are multiplied by a user-set prefactor - the correct value for the prefactor
  depends on both the timestepping scheme and the spatial discretisation, so to
  be pragmatic we tune the value by trial and error. [CFL limits associated
  with other terms in the equations could be added in a similar way if it is
  useful.]
* At each step, the timestep is allowed to increase by at most a (user-set)
  factor, to avoid large jumps that might cause numerical instability.
* There is an option to set a minimum timestep, which may be useful to push the
  simulation through initial transients where there is some numerical
  instability which would make the truncation error estimate push the timestep
  to ridiculously small values. Since we might not care about accuracy too much
  during these initial transients, it can be useful to set a minimum to stop
  the timestep getting too small (as long as the minimum is small enough that
  the simulation does not crash).

The estimates and limits just described are controlled by various tuning
parameters, described in [timestepping-input-parameters](@ref), that may need
to be set appropriately to get good performance from the adaptive timestepping
methods.  The timestep achievable may be limited by accuracy or by stability.
If the `CFL_prefactor` is set too high (or the relevant CFL limit is not being
checked) then the timestep will try to increase too high for stability - when
this happens, the step will also become inaccurate, causing timestep failures
and reducing the timestep. So the simulation should continue without crashing,
however it will be inefficient as the truncation error estimate will not 'see'
the stability limit until the limit is exceeded, resulting in a cycle of
increasing timestep followed by (probably repeated) timestep failures. The aim
should probably be to set the `CFL_limit_prefactor` and `max_increase_factor`
to the highest values that do not lead to too many timestep failures (a few
failures are OK, especially during the initial transient phase of simulations).
`step_update_prefactor` can also be decreased to use a bigger margin in the
timestep estimated from the error metric $\epsilon$ - using a smaller
`step_update_prefactor` will make the timestep smaller when it is limited by
accuracy, but this can (sometimes!) help avoid timestep failures, which _might_
decrease the total number of steps. 

Special treatment is needed for the time points where output is to be written.
When the next timestep would take the simulation time beyond the next time
where output is to be written, the timestep is set instead to take the
simulation to the output time. Then output is written and the timestep is reset
to the last full timestep value from before the output.

### Alternative algorithm for choosing the next timestep

It might turn out that the particular CFL limits that are included in the
algorithm described in [Algorithm for choosing the next timestep](@ref) are not
a complete set of the things that set the stability limit for the explicit RK
timestep. If that is the case, it may be useful to have a more generic
algorithm that can still fairly robustly choose a good timestep size, without a
large number of timestep failures. One option is described in this subsection.
For the parameters discussed, see again [timestepping-input-parameters](@ref).

If we assume that `dt` that last failed the timestep truncation error test is a
good estimate of the `dt` that is the boundary between stable and unstable
timstep values, then it makes sense to try to keep timesteps close to that (to
avoid failures), although we also want to allow the timestep to increase past
that value in case it was a bad estimate (e.g. during some sort of transient)
or because the stability limits have changed (e.g. parallel gradients in the
simulation have changed significantly). We would like to stay close to a
marginally stable (rather than marginally unstable) timestep, so take as the
estimate the last successful timestep before the most recent failed timestep
(this is stored in the code as `t_params.dt_before_last_fail[]`). When `dt` is
within a factor `last_fail_proximity_factor` of this value, we limit the
increase in timestep to `max_increase_factor_near_last_fail`, rather than
`max_increase_factor`. Suggested setup (which of course is likely to need
adjusting depending on the simulation!):
* Set `max_increase_factor_near_last_fail` to a value very close to 1, say
  1.001. This means that the timestep can only very slowly approach and exceed
  `t_params.dt_before_last_fail[]`. Setting this value closer to 1 should
  decrease the number of timestep failures.
* Set `max_increase_factor` to a relatively large value, say 1.5 or 2, so that
  when a timestep does fail, `dt` quickly recovers to a value close to the last
  successful value before the failure.
* Set `step_update_prefactor` to a relatively small value, say 0.5.
  `step_update_prefactor` controls how far `dt` is set below the value needed
  to comply with the requested tolerances. Setting a smallish value (so a large
  margin below the value that would trigger a timestep failure) seems to help -
  current guess (JTO 20/3/2024) is that: when `dt` is close to (or maybe just
  above) the value that would be unstable, the error starts to grow; with some
  margin, and with the factor by which `dt` increases limited to a small value,
  so that `dt` is at worst very marginally unstable, the truncation error
  estimate can feel the error and decrease `dt` (modestly) back to a stable
  value, before the error becomes big enough to cause a timestep failure. Once
  `dt` has been decreased (but not too much) it is again only allowed to
  increase slowly, so as long as these decreases happen often enough, `dt` can
  stay around the stability boundary without causing timestep failures.
  Decreasing this value should decrease the number of timestep failures.
* `last_fail_proximity_factor` - current guess (JTO 20/3/2024) is that the
  default value of 1.05 is reasonable. Increasing this value should decrease
  the number of timestep failures, but will also increase the number of steps
  needed before the timestep can increase past a too-low value (from a bad
  estimate, transient, changed simulation conditions, etc.).
* As a rough guideline, more than one timestep failures on average per 100
  timesteps is probably too many to be efficient, while around or less than
  this many is probably acceptable. If there are too many failures, try
  tweaking parameters as indicated above.

In at least one case JTO has been able to use this method to get a simulation
to run without imposing CFL restrictions explicitly, in a similar number of
steps as when using (well-tuned) explicit CFL restrictions.

[Input parameters](@id timestepping-input-parameters)
------------------

| Option name | Default value | Description |
| :---------- | :------------ | :---------- |
| `nstep` | 5 | `nstep*dt` is the total length of the run. For fixed-step timestepping, `nstep` is the total number of timesteps |
| `dt` | $0.00025/T$ | For fixed-step, gives the length of the timestep. For adaptive-step gives the initial guess for the timestep. $T$ in the default value is the initial temperature of the ions |
| `CFL_prefactor` | `-1.0` | Prefactor that the CFL limits from [`moment_kinetics.utils.get_minimum_CFL_z`](@ref), [`moment_kinetics.utils.get_minimum_CFL_vpa`](@ref), [`moment_kinetics.utils.get_minimum_CFL_neutral_z`](@ref), [`moment_kinetics.utils.get_minimum_CFL_neutral_vz`](@ref) are multiplied by to set the timestep limit. If no value is given, a default is set according to which timestepping scheme is chosen (see [`moment_kinetics.runge_kutta.setup_runge_kutta_coefficients!`](@ref)). |
| `nwrite` | `1` | Output of moment quantities is written every `nwrite*dt` time units. |
| `nwrite_dfns` | `nothing` | Output of distribution function quantities is written every `nwrite_dfns*dt` time units. By default distribution function quantities are written only at the beginning and end of the simulation. |
| `type` | `"SSPRK4"` | Timestepping method, see [Fixed-timestep schemes](@ref) and [Adaptive-timestep schemes](@ref). |
| `split_operators` | `false` | If true, use operator splitting. Operator splitting is currently only partially implemented. |
| `stopfile_name` | `"stop"` | Name of the file that can be created in the output directory to stop the simulation cleanly after the next output is written. |
| `steady_state_residual` | `false` | Set to `true` to print out the maximum residual ``r(t) = \frac{\left\| n(t)-n(t-\delta t)\right\| }{\delta t}`` of the density for each species at each output step |
| `converged_residual_value` | -1.0 | If `steady_state_residual = true` and `converged_residual_value` is set to a positive value, then the simulation will be stopped if all the density residuals are less than `converged_residual_value`. Note the residuals are only calculated and checked at time steps where output for moment variables is written. |
| `rtol` | `1.0e-5` | Relative tolerance used for the truncation error metric. |
| `atol` | `1.0e-12` | Absolute tolerance used for the truncation error metric. |
| `atol_upar` | `1.0e-2*rtol` | Absolute tolerance used parallel flow moment variables in the truncation error metric. This is separate from `atol` as the flow moments are expected to pass through zero somewhere, unlike distribution functions, densities, or pressures that should always be positive. |
| `step_update_prefactor` | `0.9` | When timestep is limited by accuracy (rather than something else), it is set to `step_update_prefactor` times the estimated timestep which would give an RMS error metric `$\epsilon$` of 1 at the next step. This value should always be less than 1. Smaller values give a bigger margin under the failure threshold, so may help reduce the number of timestep failures. |
| `max_increase_factor` | `1.05` | Timestep can be increased by at most this factor at each step. |
| `max_increase_factor_near_last_fail` | `Inf` | If set to finite value, replaces `max_increase_factor` when the timestep is near the last failed `dt` value (defined as within `last_fail_proximity_factor` of the last successful `dt` value before a timestep failure). If set, must be less than `max_increase_factor`. |
| `last_fail_proximity_factor` | `1.05` | Defines the range considered 'near to' the last failed `dt` value: `dt_before_last_fail/last_fail_proximity_factor < dt < dt_before_last_fail*last_fail_proximity_factor`. |
| `minimum_dt` | `0.0` | Timestep is not allowed to decrease below this value, regardless of accuracy or stability limits. |
| `maximum_dt` | `Inf` | Timestep is not allowed to increase above this value. |
| `high_precision_error_sum` | `false` | If this is set to `true`, then quad-precision values (`Float128` from the `Quadmath` package) are used to calculate the sum in the truncation error estimates. When different numbers of processes are used, the sums are calculated in different orders, so the rounding errors will be different. When adaptive timestepping is used this means that different timesteps will be used when different numbers of processes are used, so results will not be exactly the same (although they should be consistent within the timestepper tolerances and discretisation errors). When comparing 'identical' simulations run on different numbers of processes (e.g. for debugging), these differences can be inconvenient. The differences can be avoided (or at least massively reduced) by using a higher precision for the sum, so that the order of the addition operations does not matter (at least until there are so many contributions to the sum that the rounding errors reduce the precision of the quad-precision result to less than double-precision, which would take a very large number!). This feature was originally added in an attempt to make adaptive-timestepping tests give consistent results (at a level $\sim 10^{-14}$) on the CI servers. However, rounding errors change randomly on different systems (operating system, compiler, hardware, etc.), not only because of the different order of terms in the sum in the truncation error norm, so consistency is not possible between different systems even with this feature. |

Diagnostics
-----------

To help tune the settings for adaptive timestepping methods, several
diagnostics are written to the output files:
* `step_counter` is the cumulative number of time steps taken to reach each
  output.
* `dt` is the most recent timestep size at each output.
* `failure_counter` is the cumulative number of timestep failures.
* `failure_caused_by` counts the (cumulative) number of times each evolved
  variable (distribution functions or, for moment-kinetic simulations, moment
  variables) caused a timestep failure. `failure_caused_by` is a 2d array - the
  second dimension is time, the index of the first indicates the variable (see
  below for plotting of the diagnostics).
* `limit_caused_by` counts the (cumulative) number of times each factor
  (accuracy, CFL limits, maximum timestep increase factor, minimum timestep)
  set the timestep limit. `limit_caused_by` is a 2d array - the second
  dimension is time, the index of the first indicates the factor (see below for
  plotting of the diagnostics).

These diagnostics (after being converted from cumulative counts to
counts per output step) as well as the CFL limits are plotted and/or animated
by [`makie_post_processing.timestep_diagnostics`](@ref). This function will be
called when running [`makie_post_processing.makie_post_process`](@ref) if
options in the `[timestep_diagnostics]` section of the post processing input
are set: `plot=true` for plots or `animate_CFL=true` to make animations of the
CFL limits for various terms.

Developing
----------

The script `utils/calculate_rk_coeffs.jl` provides some functions to convert a
'Butcher tableau' to the `rk_coefs` array used internally in `moment_kinetics`.
To add more RK methods (adaptive or fixed-step) it may be useful to add them in
this script, to get the `rk_coefs` values, which can be copied into
[`moment_kinetics.runge_kutta.setup_runge_kutta_coefficients!`](@ref).

API
---

See [`moment_kinetics.time_advance`](@ref),
[`moment_kinetics.runge_kutta`](@ref).
