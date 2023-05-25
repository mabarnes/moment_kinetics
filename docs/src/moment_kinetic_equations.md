Moment kinetic equations
========================

The following are partial notes on the derivation of the equations being solved
by moment\_kinetics. It would be useful to expand them with more details from
the Excalibur/Neptune reports. Equation references give the report number and
equation number, e.g. (TN-04;1) is equation (1) from report TN-04.pdf.

The drift kinetic equation (DKE), marginalised over $v_{\perp}$, for ions is,
adding ionization to the form in (TN-04;1),

```math
\begin{align}
  \frac{\partial f_{i}}{\partial t}
  +v_{\|}\frac{\partial f_{i}}{\partial z}
  -\frac{e}{m}\frac{\partial\phi}{\partial z}\frac{\partial f_{i}}{\partial v_{\|}}
  & = -R_{\mathrm{in}}\left(n_{n}f_{i}-n_{i}f_{n}\right)+R_{\mathrm{ion}}n_{i}f_{n},
\end{align}
```

and for neutrals, adding ionization to (TN-04;2)
```math
\begin{align}
  \frac{\partial f_{n}}{\partial t}
  +v_{\|}\frac{\partial f_{n}}{\partial z}
  & = -R_{\mathrm{in}}\left(n_{i}f_{n}-n_{n}f_{i}\right)-R_{\mathrm{ion}}n_{i}f_{n}
\end{align}
```

Using the normalizations (TN04;5-11)

```math
\begin{align}
  \tilde{f}_{s} & \doteq f_{s}\frac{c_{s}\sqrt{\pi}}{N_{e}}\\
  \tilde{t} & \doteq t\frac{c_{s}}{L_{z}}\\
  \tilde{z} & \doteq\frac{z}{L_{z}}\\
  \tilde{v}_{\|} & \doteq\frac{v_{\|}}{c_{s}}\\
  \tilde{n}_{s} & \doteq\frac{n_{s}}{N_{e}}\\
  \tilde{\phi} & \doteq\frac{e\phi}{T_{e}}\\
  \tilde{R}_{\mathrm{in}} & \doteq R_{\mathrm{in}}\frac{N_{e}L_{z}}{c_{s}}\\
  \tilde{R}_{\mathrm{ion}} & \doteq R_{\mathrm{ion}}\frac{N_{e}L_{z}}{c_{s}}
\end{align}
```

with $c_{s}\doteq\sqrt{2T_{e}/m_{s}}$ where $L_{z}$, $N_{e}$ and $T_{e}$ are
constant reference parameters, the ion DKE is

```math
\begin{align}
  \frac{\partial\tilde{f}_{i}}{\partial\tilde{t}}
  + \tilde{v}_{\|}\frac{\partial\tilde{f}_{i}}{\partial\tilde{z}}
  - \frac{1}{2}\frac{\partial\tilde{\phi}}{\partial\tilde{z}}
    \frac{\partial\tilde{f}_{i}}{\partial\tilde{v}_{\|}}
  & = -\tilde{R}_{in}\left(\tilde{n}_{n}\tilde{f}_{i}-\tilde{n}_{i}\tilde{f}_{n}\right)
      + \tilde{R}_{\mathrm{ion}}\tilde{n}_{i}\tilde{f}_{n}
\end{align}
```

and the neutral DKE is

```math
\begin{align}
  \frac{\partial\tilde{f}_{n}}{\partial\tilde{t}}
  + v_{\|}\frac{\partial\tilde{f}_{n}}{\partial\tilde{z}}
  & = -\tilde{R}_{in}\left(\tilde{n}_{i}\tilde{f}_{n}-\tilde{n}_{n}\tilde{f}_{i}\right)
      - \tilde{R}_{\mathrm{ion}}\tilde{n}_{i}\tilde{f}_{n}
\end{align}
```

Moment equations
----------------

Recalling the definitions (TN-04;15,29,63-66), but writing the integral in the
energy equation over $\tilde{v}_{\|}$ instead of $w_{\|}$,

```math
\begin{align}
  \tilde{n}_{s}
  & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{f}_{s}\\
  %
  \tilde{n}_{s}\tilde{u}_{s}
  & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}\tilde{f}_{s}\\
  %
  \tilde{p}_{\|,s}
  & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\left(\tilde{v}_{\|}
      - \tilde{u}_{s}\right)^{2}\tilde{f}_{s}
    = \int d\tilde{v}_{\|}\tilde{v}_{\|}^{2}\tilde{f}_{s}
      - \tilde{n}_{s}\tilde{u}_{s}^{2}\\
  %
  \tilde{q}_{\|,s}
  & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}
      \left(\tilde{v}_{\|}-\tilde{u}_{s}\right)^{3}\tilde{f}_{s}
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \tilde{q}_{\|,s}
   & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}^{3}\tilde{f}_{s}
       - 3\tilde{u}_{s}\frac{1}{\sqrt{\pi}}\int dv_{\|}v_{\|}^{2}f_{s}
       + 3u_{s}^{2}\frac{1}{\sqrt{\pi}}\int dv_{\|}v_{\|}f_{s}
       - u_{s}^{3}\frac{1}{\sqrt{\pi}}\int dv_{\|}f_{s} \\
  %
   & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}^{3}\tilde{f}_{s}
       - 3\tilde{u}_{s}\left(\tilde{p}_{\|,s}+\tilde{n}_{s}\tilde{u}_{s}^{2}\right)
       + 3\tilde{u}_{s}^{2}\tilde{n}_{s}\tilde{u}_{s}-\tilde{u}_{s}^{3}\tilde{n}_{s}
\end{align*}
```
```@raw html
</details>
```

```math
\begin{align}
  \tilde{q}_{\|,s}
    & = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}^{3}\tilde{f}_{s}
         - 3\tilde{u}_{s}\tilde{p}_{\|,s}
         - \tilde{n}_{s}\tilde{u}_{s}^{3}
\end{align}
```

we can take moments of the kinetic equations to give moment equations (dropping
tildes from here on)

#### Ions

```math
\begin{align}
  \frac{\partial n_{i}}{\partial t}+\frac{\partial\left(n_{i}u_{i}\right)}{\partial z}
  & = -R_{in}\left(n_{n}n_{i}-n_{i}n_{n}\right)+R_{\mathrm{ion}}n_{i}n_{n} \\
  %
  & = R_{\mathrm{ion}}n_{i}n_{n}
\end{align}
```

```math
\begin{align}
  \frac{\partial\left(n_{i}u_{i}\right)}{\partial t} + \frac{\partial\left(p_{\|,i}
  + n_{i}u_{i}^{2}\right)}{\partial z} + \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  & = -R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
      + R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  n_{i}\frac{\partial u_{i}}{\partial t} + u_{i}\frac{\partial n_{i}}{\partial t}
  + \frac{\partial p_{\|,i}}{\partial z}
  + u_{i}\frac{\partial\left(n_{i}u_{i}\right)}{\partial z}
  + n_{i}u_{i}\frac{\partial u_{i}}{\partial z}
  + \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  & = -R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
      + R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
  %
  n_{i}\frac{\partial u_{i}}{\partial t} + u_{i}\left(R_{\mathrm{ion}}n_{i}n_{n}\right)
  + \frac{\partial p_{\|,i}}{\partial z} + n_{i}u_{i}\frac{\partial u_{i}}{\partial z}
  + \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  & = -R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
      + R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  \frac{\partial u_{i}}{\partial t} + \frac{1}{n_{i}}\frac{\partial p_{\|,i}}{\partial z}
  + u_{i}\frac{\partial u_{i}}{\partial z} + \frac{1}{2}\frac{\partial\phi}{\partial z}
  & = -R_{in}n_{n}\left(u_{i}-u_{n}\right)
      + R_{\mathrm{ion}}\frac{n_{i}n_{n}}{n_{s}}\left(u_{n}-u_{i}\right)
\end{align}
```

```math
\begin{align}
  & \frac{\partial\left(p_{\|,i} + n_{i}u_{i}^{2}\right)}{\partial t}
    + \frac{\partial\left(q_{\|,i} + 3u_{i}p_{\|,i}
    + n_{i}u_{i}^{3}\right)}{\partial z} + \frac{\partial\phi}{\partial z}n_{i}u_{i} \\
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n}+n_{n}u_{n}^{2}\right) \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \frac{\partial p_{\|,i}}{\partial t}
  + \frac{1}{n_{i}}\frac{\partial\left(n_{i}u_{i}\right)^{2}}{\partial t}
  - \frac{\left(n_{i}u_{i}\right)^{2}}{n_{i}^{2}}\frac{\partial n_{i}}{\partial t}
  + \frac{\partial\left(q_{\|,i} + 3u_{i}p_{\|,i}
  + n_{i}u_{i}^{3}\right)}{\partial z} + \frac{\partial\phi}{\partial z}n_{i}u_{i}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
                   - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
  %
  \frac{p_{\|,i}}{\partial t} + 2u_{i}\frac{\partial n_{i}u_{i}}{\partial t}
  - u_{i}^{2}\frac{\partial n_{i}}{\partial t} + \frac{\partial\left(q_{\|,i}
  + 3u_{i}p_{\|,i} + n_{i}u_{i}^{3}\right)}{\partial z}
  + \frac{\partial\phi}{\partial z}n_{i}u_{i}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
  %
  \frac{\partial p_{\|,i}}{\partial t} + 2u_{i}\left(-\frac{\partial p_{\|,i}}{\partial z}
  - \frac{\partial\left(n_{i}u_{i}^{2}\right)}{\partial z}
  - \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  - R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
  + R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right) \\
  -u_{i}^{2}\left(-\frac{\partial\left(n_{i}u_{i}\right)}{\partial z}
  + R_{\mathrm{ion}}n_{i}n_{n}\right) + \frac{\partial q_{\|,i}}{\partial z}
  + \frac{\partial\left(3u_{i}p_{\|,i}\right)}{\partial z}
  + \frac{\partial\left(n_{i}u_{i}^{3}\right)}{\partial z}
  + \frac{\partial\phi}{\partial z}n_{i}u_{i}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
  %
  \frac{\partial p_{\|,i}}{\partial t} + u_{i}\frac{\partial p_{\|,i}}{\partial z}
  + 3p_{\|,i}\frac{\partial u_{i}}{\partial z} + \frac{\partial q_{\|,i}}{\partial z}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right) - n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2}\right) - 2u_{i}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)\right) \\
      & \quad + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2} + n_{n}u_{i}^{2}
      - 2n_{n}u_{i}u_{n}\right) \\
  %
  \frac{\partial p_{\|,i}}{\partial t} + u_{i}\frac{\partial p_{\|,i}}{\partial z}
  + 3p_{\|,i}\frac{\partial u_{i}}{\partial z} + \frac{\partial q_{\|,i}}{\partial z}
  & = -R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n} - n_{i}n_{n}\left(u_{i}^{2} + u_{n}^{2}
      - 2u_{i}u_{n}\right)\right) + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}\left(u_{n}
      - u_{i}\right)^{2}\right) \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  & \frac{\partial p_{\|,i}}{\partial t} + u_{i}\frac{\partial p_{\|,i}}{\partial z}
    + 3p_{\|,i}\frac{\partial u_{i}}{\partial z} + \frac{\partial q_{\|,i}}{\partial z} \\
  & = -R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n}
      - n_{i}n_{n}\left(u_{i} - u_{n}\right)^{2}\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}\left(u_{n} - u_{i}\right)^{2}\right)
\end{align}
```

##### Implemented forms

The continuity equation for ions is implemented by
[`continuity_equation_single_species!`](@ref
moment_kinetics.continuity.continuity_equation_single_species!) in the slightly
rearranged form

```math
\begin{align}
  \frac{\partial n_{i}}{\partial t} & = - u_{i}\frac{\partial n_{i}}{\partial z}
                                        - n_{i}\frac{\partial u_{i}}{\partial z}
                                        + R_{\mathrm{ion}} n_{i} n_{n}
\end{align}
```

The momentum equation for ions is implemented by [`force_balance!`](@ref
moment_kinetics.force_balance.force_balance!) in the rearranged form

```math
\begin{align}
  \frac{\partial n_{i}u_{i}}{\partial t} & = - \frac{\partial p_{\|,i}}{\partial z}
                                             - u_{i}^2 \frac{\partial n_{i}}{\partial z}
                                             - 2 n_{i} u_{i} \frac{\partial u_{i}}{\partial z}
                                             - \frac{1}{2} \frac{\partial \phi}{\partial z} n_{i}
                                             + R_{in} n_{i} n_{n} (u_{n} - u_{i})
                                             + R_{\mathrm{ion}} n_{i} n_{n} u_{n}
\end{align}
```

The energy equation for ions is implemented by [`energy_equation!`](@ref
moment_kinetics.energy_equation.energy_equation!) as

```math
\begin{align}
  \frac{\partial p_{\|,i}}{\partial t}
    & = - u_{i} \frac{\partial p_{\|,i}}{\partial z}
        - \frac{\partial q_{\|,i}}{\partial z}
        - 3 p_{\|,i} \frac{\partial u_{i}}{\partial z}
        - R_{in} \left(
                       n_{n} p_{\|,i} - n_{i} p_{\|,n}
                       - n_{i} n_{n} \left(u_{i} - u_{n}\right)^2
                 \right)
        + R_{\mathrm{ion}} n_{i} \left(
                                       p_{\|,n} + n_{n} \left(u_{i} - u_{n}\right)^2
                                 \right)
\end{align}
```

#### Neutrals

```math
\begin{align}
  \frac{\partial n_{n}}{\partial t} + \frac{\partial\left(n_{n}u_{n}\right)}{\partial z}
  & = -R_{i}\left(n_{i}n_{n} - n_{n}n_{i}\right) - R_{\mathrm{ion}}n_{i}n_{n} \\
  %
  & =-R_{\mathrm{ion}}n_{i}n_{n}
\end{align}
```

```math
\begin{align}
  \frac{\partial\left(n_{n}u_{n}\right)}{\partial t}
  + \frac{\partial\left(p_{\|,n} + n_{n}u_{n}^{2}\right)}{\partial z}
  & = -R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
      - R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  n_{n}\frac{\partial u_{n}}{\partial t} + u_{n}\frac{\partial n_{n}}{\partial t}
  + \frac{\partial p_{\|,n}}{\partial z}
  + u_{n}\frac{\partial\left(n_{n}u_{n}\right)}{\partial z}
  + n_{n}u_{n}\frac{\partial u_{n}}{\partial z}
  & = -R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
      - R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
  %
  n_{n}\frac{\partial u_{n}}{\partial t}
  + u_{n}\left(-R_{\mathrm{ion}}n_{i}n_{n}\right) + \frac{\partial p_{\|,n}}{\partial z}
  + n_{n}u_{s}\frac{\partial u_{n}}{\partial z}
  & = -R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
      - R_{\mathrm{ion}}n_{i}n_{n}u_{n} \\
\end{align}
```

```@raw html
</details>
```

```math
\begin{align}
  \frac{\partial u_{n}}{\partial t} + \frac{1}{n_{n}}\frac{\partial p_{\|,n}}{\partial z}
  + u_{n}\frac{\partial u_{n}}{\partial z}
  & = -R_{in}n_{i}\left(u_{n} - u_{i}\right)
\end{align}
```

```math
\begin{align}
  & \frac{\partial\left(p_{\|,n} + n_{n}u_{n}^{2}\right)}{\partial t}
    + \frac{\partial\left(q_{\|,n} + 3u_{n}p_{\|,n}
    + n_{n}u_{n}^{3}\right)}{\partial z} + q_{n}\frac{\partial\phi}{\partial z}n_{n}u_{n} \\
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right)\right)
      - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \frac{\partial p_{\|,n}}{\partial t}
  + \frac{1}{n_{n}}\frac{\partial\left(n_{n}u_{n}\right)^{2}}{\partial t}
  - \frac{\left(n_{n}u_{n}\right)^{2}}{n_{n}^{2}}\frac{\partial n_{n}}{\partial t}
  + \frac{\partial\left(q_{\|,n} + 3u_{n}p_{\|,n} + n_{n}u_{n}^{3}\right)}{\partial z}
  + q_{n}\frac{\partial\phi}{\partial z}n_{n}u_{n}
  & =-R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right)\right)
      - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
  %
  \frac{\partial p_{\|,n}}{\partial t} + 2u_{n}\frac{\partial n_{n}u_{n}}{\partial t}
  - u_{n}^{2}\frac{\partial n_{n}}{\partial t} + \frac{\partial\left(q_{\|,n}
  + 3u_{n}p_{\|,n} + n_{n}u_{n}^{3}\right)}{\partial z}
  + q_{n}\frac{\partial\phi}{\partial z}n_{n}u_{n}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right)\right) - R_{\mathrm{ion}}n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2}\right) \\
  %
  \frac{\partial p_{\|,n}}{\partial t}
  + 2u_{n}\left(-\frac{\partial p_{\|,n}}{\partial z}
  - \frac{\partial\left(n_{n}u_{n}^{2}\right)}{\partial z}
  - \frac{q_{n}}{2}\frac{\partial\phi}{\partial z}n_{n}
  - R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
  - R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right) \\
  - u_{n}^{2}\left(-\frac{\partial\left(n_{n}u_{n}\right)}{\partial z}
  - R_{\mathrm{ion}}n_{i}n_{n}\right) + \frac{\partial q_{\|,n}}{\partial z}
  + \frac{\partial\left(3u_{n}p_{\|,n}\right)}{\partial z}
  + \frac{\partial\left(n_{n}u_{n}^{3}\right)}{\partial z}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
  + n_{i}u_{i}^{2}\right)\right)
  - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) \\
  %
  \frac{\partial p_{\|,n}}{\partial t} + u_{n}\frac{\partial p_{\|,n}}{\partial z}
  + 3p_{\|,n}\frac{\partial u_{n}}{\partial z} + \frac{\partial q_{\|,n}}{\partial z}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right) - 2u_{n}\left(n_{i}n_{n}u_{n}
      - n_{n}n_{i}u_{i}\right)\right) - R_{\mathrm{ion}}n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2} + n_{n}u_{n}^{2} - 2n_{n}u_{n}u_{n}\right) \\
  %
  \frac{\partial p_{\|,n}}{\partial t} + u_{n}\frac{\partial p_{\|,n}}{\partial z}
  + 3p_{\|,n}\frac{\partial u_{n}}{\partial z} + \frac{\partial q_{\|,n}}{\partial z}
  & = -R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i} - n_{n}n_{i}\left(u_{n}^{2} + u_{i}^{2}
      - 2u_{n}u_{i}\right)\right) - R_{\mathrm{ion}}n_{i}p_{\|,n} \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  & \frac{\partial p_{\|,n}}{\partial t} + u_{n}\frac{\partial p_{\|,n}}{\partial z}
    + 3p_{\|,n}\frac{\partial u_{n}}{\partial z} + \frac{\partial q_{\|,n}}{\partial z} \\
  & = -R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i}
      - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right) - R_{\mathrm{ion}}n_{i}p_{\|,n}
\end{align}
```

##### Implemented forms

The continuity equation for neutrals is implemented by
[`continuity_equation_single_species!`](@ref
moment_kinetics.continuity.continuity_equation_single_species!) in the slightly
rearranged form

```math
\begin{align}
  \frac{\partial n_{n}}{\partial t} & = - u_{n}\frac{\partial n_{n}}{\partial z}
                                        - n_{n}\frac{\partial u_{n}}{\partial z}
                                        - R_{\mathrm{ion}} n_{n} n_{i}
\end{align}
```

The momentum equation for neutrals is implemented by [`force_balance!`](@ref
moment_kinetics.force_balance.force_balance!) in the rearranged form

```math
\begin{align}
  \frac{\partial n_{n}u_{n}}{\partial t} & = - \frac{\partial p_{\|,n}}{\partial z}
                                             - u_{n}^2 \frac{\partial n_{n}}{\partial z}
                                             - 2 n_{n} u_{n} \frac{\partial u_{n}}{\partial z}
                                             + R_{in} n_{n} n_{i} (u_{i} - u_{n})
                                             - R_{\mathrm{ion}} n_{i} n_{n} u_{n}
\end{align}
```

The energy equation for neutrals is implemented by [`energy_equation!`](@ref
moment_kinetics.energy_equation.energy_equation!) as

```math
\begin{align}
  \frac{\partial p_{\|,n}}{\partial t}
    & = - u_{n} \frac{\partial p_{\|,n}}{\partial z}
        - \frac{\partial q_{\|,n}}{\partial z}
        - 3 p_{\|,n} \frac{\partial u_{n}}{\partial z}
        - R_{in} \left(
                       n_{i} p_{\|,n} - n_{n} p_{\|,i}
                       - n_{n} n_{i} \left(u_{n} - u_{i}\right)^2
                 \right)
        - R_{\mathrm{ion}} n_{i} p_{\|,n}
\end{align}
```

Kinetic equation
----------------

For the moment-kinetic equation for the normalized distribution function

```math
\begin{align}
  g_{s}(w_{\|,s}) & = \frac{v_{\mathrm{th},s}}{n_{s}}f_{s}(v_{\|}(w_{\|,s}))
\end{align}
```

we transform to the normalized velocity coordinate

```math
\begin{align}
  w_{\|,s} & = \frac{v_{\|} - u_{s}}{v_{\mathrm{th},s}}
\end{align}
```

The derivatives transform as

```math
\begin{align}
  \left.\frac{\partial f_{s}}{\partial t}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial t}\right|_{z,w\|}
               - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
               - \frac{w_{\|,s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  \left.\frac{\partial f_{s}}{\partial z}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial z}\right|_{z,w\|}
               - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
               - \frac{w_{\|,s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  \left.\frac{\partial f_{s}}{\partial v_{\|}}\right|_{z,v\|}
  & \rightarrow\frac{1}{v_{\mathrm{th},s}}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
\end{align}
```

We use an energy equation that evolves $p_{\|,s}$ not $v_{\mathrm{th},s}$, so
use

```math
\begin{align}
  v_{\mathrm{th},s}^{2} & = 2\frac{p_{\|,s}}{n_{s}} \\
  %
  \Rightarrow v_{\mathrm{th},s}\frac{\partial v_{\mathrm{th},s}}{\partial t}
  & = \frac{1}{n_{s}}\frac{\partial p_{\|,s}}{\partial t}
      - \frac{p_{\|,s}}{n_{s}^{2}}\frac{\partial n_{s}}{\partial t}\\
  %
  v_{\mathrm{th},s}\frac{\partial v_{\mathrm{th},s}}{\partial z}
  & = \frac{1}{n_{s}}\frac{\partial p_{\|,s}}{\partial z}
      - \frac{p_{\|,s}}{n_{s}^{2}}\frac{\partial n_{s}}{\partial z}
\end{align}
```

to convert the transformations above to

```math
\begin{align}
  \left.\frac{\partial f_{s}}{\partial t}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial t}\right|_{z,w\|}
    - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
    - \frac{w_{\|,s}}{v_{\mathrm{th},s}^{2}}\left(\frac{1}{n_{s}}\frac{\partial p_{\|,s}}{\partial t}
    - \frac{p_{\|,s}}{n_{s}^{2}}\frac{\partial n_{s}}{\partial t}\right)\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  & = \left.\frac{\partial f_{s}}{\partial t}\right|_{z,w\|}
      - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
      - \frac{w_{\|,s}}{2}\left(\frac{1}{p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
      - \frac{1}{n_{s}}\frac{\partial n_{s}}{\partial t}\right)\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  \left.\frac{\partial f_{s}}{\partial z}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial z}\right|_{z,w\|}
    - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
    - \frac{w_{\|,s}}{v_{\mathrm{th},s}^{2}}\left(\frac{1}{n_{s}}\frac{\partial p_{\|,s}}{\partial z}
    - \frac{p_{\|,s}}{n_{s}^{2}}\frac{\partial n_{s}}{\partial z}\right)\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  & = \left.\frac{\partial f_{s}}{\partial z}\right|_{z,w\|}
      - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
      - \frac{w_{\|,s}}{2}\left(\frac{1}{p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z}
      - \frac{1}{n_{s}}\frac{\partial n_{s}}{\partial z}\right)\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}\\
  %
  \left.\frac{\partial f_{s}}{\partial v_{\|}}\right|_{z,v\|}
  & \rightarrow\frac{1}{v_{\mathrm{th},s}}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,w\|}
\end{align}
```

Using these transformations gives the ion DKE in a form similar to (TN-04;55)
(but writing out $\dot{w}_{\|}$ in full here, and not using the moment
equations for the moment)

```math
\begin{align}
  & \frac{\partial f_{i}}{\partial t}
    - \frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial t}\frac{\partial f_{i}}{\partial w_{\|,i}}
    - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial t}
    - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial t}\right)\frac{\partial f_{i}}{\partial w_{\|,i}} \\
  & + \left(v_{\mathrm{th},i}w_{\|,i} + u_{i}\right)\left(\frac{\partial f_{i}}{\partial z}
    - \frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial z}\frac{\partial f_{i}}{\partial w_{\|,i}}
    - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial z}
    - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial z}\right)\frac{\partial f_{i}}{\partial w_{\|,i}}\right) \\
  & - \frac{1}{2v_{\mathrm{th},i}}\frac{\partial\phi}{\partial z}\frac{\partial f_{i}}{\partial w_{\|,i}} \\
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n}
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \frac{\partial f_{i}}{\partial t} + \left(v_{\mathrm{th},i}w_{\|,i}
  + u_{i}\right)\frac{\partial f_{i}}{\partial z}
  - \frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial t}\frac{\partial f_{i}}{\partial w_{\|,i}}
  - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial t}
  - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial t}\right)\frac{\partial f_{i}}{\partial w_{\|,i}}
  + \left(v_{\mathrm{th},i}w_{\|,i}
  + u_{i}\right)\left(-\frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial z}\frac{\partial f_{i}}{\partial w_{\|,i}}
  - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial z}
  - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial z}\right)\frac{\partial f_{i}}{\partial w_{\|,i}}\right)
  - \frac{1}{2v_{\mathrm{th},i}}\frac{\partial\phi}{\partial z}\frac{\partial f_{i}}{\partial w_{\|,i}}
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} \\
  %
  \frac{\partial f_{i}}{\partial t} + \left(v_{\mathrm{th},i}w_{\|,i}
  + u_{i}\right)\frac{\partial f_{i}}{\partial z}
  + \left[-\frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial t}
  - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial t}
  - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial t}\right)
  + \left(v_{\mathrm{th},i}w_{\|,i}
  + u_{i}\right)\left(-\frac{1}{v_{\mathrm{th},i}}\frac{\partial u_{i}}{\partial z}
  - \frac{w_{\|,i}}{2}\left(\frac{1}{p_{\|,i}}\frac{\partial p_{\|,i}}{\partial z}
  - \frac{1}{n_{i}}\frac{\partial n_{i}}{\partial z}\right)\right)
  - \frac{1}{2v_{\mathrm{th},i}}\frac{\partial\phi}{\partial z}\right]\frac{\partial f_{i}}{\partial w_{\|,i}}
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  & \frac{\partial f_{i}}{\partial t} + \left(v_{\mathrm{th},i}w_{\|,i}
    + u_{i}\right)\frac{\partial f_{i}}{\partial z} \\
  & + \left[-\frac{1}{v_{\mathrm{th},i}}\left(\frac{\partial u_{i}}{\partial t}
    + \left(v_{\mathrm{th},i}w_{\|,i} + u_{i}\right)\frac{\partial u_{i}}{\partial z}
    + \frac{1}{2}\frac{\partial\phi}{\partial z}\right)\right. \\
  & \qquad - \frac{w_{\|,i}}{2}\frac{1}{p_{\|,i}}\left(\frac{\partial p_{\|,i}}{\partial t}
          + \left(v_{\mathrm{th},i}w_{\|,i} + u_{i}\right)\frac{\partial p_{\|,i}}{\partial z}\right) \\
  & \qquad + \frac{w_{\|,i}}{2}\frac{1}{n_{i}}\left(\frac{\partial n_{i}}{\partial t}
          + \left(v_{\mathrm{th},i}w_{\|,i}
          + \left.u_{i}\right)\frac{\partial n_{i}}{\partial z}\right)\right]\frac{\partial f_{i}}{\partial w_{\|,i}} \\
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n}
\end{align}
```

and the neutral DKE

```math
\begin{align}
  & \frac{\partial f_{n}}{\partial t}
    - \frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial t}\frac{\partial f_{n}}{\partial w_{\|,n}}
    - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial t}
    - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial t}\right)\frac{\partial f_{n}}{\partial w_{\|,n}} \\
  & + \left(v_{\mathrm{th},n}w_{\|,n} + u_{n}\right)\left(\frac{\partial f_{n}}{\partial z}
    - \frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial z}\frac{\partial f_{n}}{\partial w_{\|,n}}
    - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial z}
    - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)\frac{\partial f_{n}}{\partial w_{\|,n}}\right) \\
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \frac{\partial f_{n}}{\partial t} + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\frac{\partial f_{n}}{\partial z}
  - \frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial t}\frac{\partial f_{n}}{\partial w_{\|,n}}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial t}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial t}\right)\frac{\partial f_{n}}{\partial w_{\|,n}}
  + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\left(-\frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial z}\frac{\partial f_{n}}{\partial w_{\|,n}}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial z}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)\frac{\partial f_{n}}{\partial w_{\|,n}}\right)
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} \\
  %
  \frac{\partial f_{n}}{\partial t} + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\frac{\partial f_{n}}{\partial z}
  + \left[-\frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial t}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial t}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial t}\right) + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\left(-\frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial z}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial z}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)\right)\right]\frac{\partial f_{n}}{\partial w_{\|,n}}
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  & \frac{\partial f_{n}}{\partial t} + \left(v_{\mathrm{th},n}w_{\|,n}
    + u_{n}\right)\frac{\partial f_{n}}{\partial z} \\
  & + \left[-\frac{1}{v_{\mathrm{th},n}}\left(\frac{\partial u_{n}}{\partial t}
    + \left(v_{\mathrm{th},n}w_{\|,n}+u_{n}\right)\frac{\partial u_{n}}{\partial z}\right)\right. \\
  & \qquad - \frac{w_{\|,n}}{2}\frac{1}{p_{\|,n}}\left(\frac{\partial p_{\|,n}}{\partial t}
           + \left(v_{\mathrm{th},n}w_{\|,n} + u_{n}\right)\frac{\partial p_{\|,n}}{\partial z}\right) \\
  & \qquad + \left.\frac{w_{\|,n}}{2}\frac{1}{n_{n}}\left(\frac{\partial n_{n}}{\partial t}
           + \left(v_{\mathrm{th},n}w_{\|,n}
           + u_{n}\right)\frac{\partial n_{n}}{\partial z}\right)\right]\frac{\partial f_{n}}{\partial w_{\|,n}} \\
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n}
\end{align}
```

We also normalise $f$ and write the DKEs for

```math
\begin{align}
  g_{s} & =\frac{v_{\mathrm{th,s}}}{n_{s}}f_{s} \\
  %
  \Rightarrow\frac{\partial f_{s}}{\partial t}
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial t}
  + \frac{g_{s}}{v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t}
  - \frac{n_{s}g_{s}}{v_{\mathrm{th},s}^{2}}\frac{\partial v_{\mathrm{th},s}}{\partial t} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \frac{\partial f_{s}}{\partial t}
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial t}
      + \frac{g_{s}}{v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t}
      - \frac{n_{s}g_{s}}{v_{\mathrm{th},s}^{3}}\left(\frac{1}{n_{s}}\frac{\partial p_{\|,s}}{\partial t}
      - \frac{p_{\|,s}}{n_{s}^{2}}\frac{\partial n_{s}}{\partial t}\right) \\
  %
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial t}
      + \frac{g_{s}}{v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t}
      - \frac{g_{s}n_{s}}{2v_{\mathrm{th},s}p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
      + \frac{g_{s}}{2v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t} \\
\end{align*}
```

```@raw html
</details>
```

```math
\begin{align}
  \frac{\partial f_{s}}{\partial t}
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial t}
      + \frac{3g_{s}}{2v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t}
      - \frac{g_{s}n_{s}}{2v_{\mathrm{th},s}p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  %
  \frac{\partial f_{s}}{\partial w_{\|,s}}
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial w_{\|,s}},
\end{align}
```

For brevity, do the following manipulations for $g_{s}$ rather than for ions
and neutrals separately by using $q_{i}=1$, $q_{n}=0$ and with the $+$'ve sign
for the ion DKE and $-$'ve sign for the neutral DKE.

```math
\begin{align}
  & \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial t}
  + \frac{3g_{s}}{2v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}n_{s}}{2v_{\mathrm{th},s}p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial f_{s}}{\partial z} \\
  & + \left[-\frac{1}{v_{\mathrm{th},s}}\left(\frac{\partial u_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial u_{s}}{\partial z}
  + \frac{q_{s}}{2}\frac{\partial\phi}{\partial z}\right)\right. \\
  & \qquad - \frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \qquad + \left.\frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)\right]\frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}\left(n_{s'}\frac{n_{s}}{v_{\mathrm{th},s}}g_{s}
      - n_{s}\frac{n_{s'}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  & + \left[-\frac{1}{v_{\mathrm{th},s}}\left(\frac{\partial u_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial u_{s}}{\partial z}
  + \frac{q_{s}}{2}\frac{\partial\phi}{\partial z}\right)
  - \frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial p_{\|,s}}{\partial z}\right)
  + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
  %
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  & + \left[-\frac{1}{v_{\mathrm{th},s}}\left(\frac{n_{s}}{n_{s}}\frac{\partial u_{s}}{\partial t}
  + \frac{n_{s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial u_{s}}{\partial z}
  + \frac{u_{s}}{n_{s}}\left(\frac{\partial n}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial n}{\partial z}\right)
  + \frac{q_{s}}{2}\frac{\partial\phi}{\partial z}\right)
  + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\frac{\partial n}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial n}{\partial z}\right)
  - \frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial p_{\|,s}}{\partial z}\right)
  + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
\end{align}
```

```@raw html
</details>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(\frac{\partial n_{s}u_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\left(n_{s}\frac{\partial u_{s}}{\partial z}
  + u_{s}\frac{\partial n_{s}}{\partial z}\right)
  + \frac{q_{s}}{2}n_{s}\frac{\partial\phi}{\partial z}\right)\right. \\
  & \qquad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)
  - \frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \qquad \left.+ \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
\end{align}
```

So then if we use the moment equations we can rewrite the DKE as

```math
\begin{align}
  & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(\frac{\partial n_{s}u_{s}}{\partial t}
  + u_{s}\left(n_{s}\frac{\partial u_{s}}{\partial z}
  + u_{s}\frac{\partial n_{s}}{\partial z}\right)
  - \frac{1}{2}n_{s}E_{\|}
  + v_{\mathrm{th},s}w_{\|,s}\left(n_{s}\frac{\partial u_{s}}{\partial z}
  + u_{s}\frac{\partial n_{s}}{\partial z}\right)\right)\right. \\
  & \qquad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\frac{\partial n_{s}}{\partial t}
  + u_{s}\frac{\partial n_{s}}{\partial z}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right) \\
  & \qquad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + u_{s}\frac{\partial p_{\|,s}}{\partial z}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \qquad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + u_{s}\frac{\partial n_{s}}{\partial z}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}
  - u_{s}\frac{\partial n_{s}}{\partial z}
  - n_{s}\frac{\partial u_{s}}{\partial z}\right) \\
  & -\frac{g_{s}}{2p_{\|,s}}\left(-u_{s}\frac{\partial p_{\|,s}}{\partial z}
  - \frac{\partial q_{\|,s}}{\partial z}
  - 3p_{\|,s}\frac{\partial u_{s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n} - u_{s}\right)^{2}\right)\right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\underbrace{\cancel{n_{s}u_{s}\frac{\partial u_{s}}{\partial z}}}_{A}
  - \frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}u_{n}
  + v_{\mathrm{th},s}w_{\|,s}\left(\underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{B}
  + \underbrace{\cancel{u_{s}\frac{\partial n_{s}}{\partial z}}}_{C}\right)\right)\right. \\
  & \quad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}
  - \underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{A}
  + \underbrace{\cancel{v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}}}_{C}\right) \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - \underbrace{\cancel{3p_{\|,s}\frac{\partial u_{s}}{\partial z}}}_{B}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n}
  - u_{s}\right)^{2}\right) + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}
  - \underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{B}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
  %
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}
  - u_{s}\frac{\partial n_{s}}{\partial z} - n_{s}\frac{\partial u_{s}}{\partial z}\right) \\
  & -\frac{g_{s}}{2p_{\|,s}}\left(-u_{s}\frac{\partial p_{\|,s}}{\partial z}
  - \frac{\partial q_{\|,s}}{\partial z} - 3p_{\|,s}\frac{\partial u_{s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n} - u_{s}\right)^{2}\right)\right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)\pm R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right)\right. \\
  & \quad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}\right) \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n}
  - u_{s}\right)^{2}\right) + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
  %
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + g_{s}\left(\pm\frac{3}{2}R_{\mathrm{ion}}n_{i}\frac{n_{n}}{n_{s}}
  - \frac{3u_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial z}\right) \\
  & + g_{s}\left(\frac{u_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z}
  + \frac{1}{2p_{\|,s}}\frac{\partial q_{\|,s}}{\partial z}
  + \frac{1}{2p_{\|,s}}R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \mp\frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,s}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{s}\right)^{2}\right)\right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{s}\right)\right)\right. \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\mp\frac{w_{\|,s}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,s}}
  - \frac{n_{n}}{n_{s}} + \frac{n_{n}}{p_{\|,s}}\left(u_{n} - u_{s}\right)^{2}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} \\
\end{align}
```

and finally using

```math
\begin{align}
  \frac{u_{s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial z}
  & =u_{s}\sqrt{\frac{n_{s}}{p_{\|,s}}}\frac{\partial}{\partial z}\sqrt{\frac{p_{\|,s}}{n_{s}}} \\
  & = \frac{u_{s}}{2}\left(\frac{1}{p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z}
      - \frac{1}{n_{s}}\frac{\partial n_{s}}{\partial z}\right)
\end{align}
```

gives

```@raw html
</details>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \frac{v_{\mathrm{th},s}}{n_{s}}\left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial f_{s}}{\partial z}
  + \left(\pm\frac{3}{2}R_{\mathrm{ion}}n_{i}\frac{n_{n}}{n_{s}}
  - \frac{u_{s}}{n_{s}}\frac{\partial n_{s}}{\partial z}\right)g_{s} \\
  & + \left(\frac{u_{s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial z}
  + \frac{1}{2p_{\|,s}}\frac{\partial q_{\|,s}}{\partial z}\right. \\
  & \qquad + \frac{1}{2p_{\|,s}}R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right) \\
  & \qquad \left.\mp\frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,s}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{s}\right)^{2}\right)\right)g_{s} \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{s}\right)\right)\right. \\
  & \qquad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \qquad\mp\frac{w_{\|,s}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,s}}
  - \frac{n_{n}}{n_{s}} + \frac{n_{n}}{p_{\|,s}}\left(u_{n} - u_{s}\right)^{2}\right) \\
  & \qquad\left. + \frac{w_{\|,s}^{2}}{2}\frac{v_{\mathrm{th},s}}{n_{s}}\frac{\partial n_{s}}{\partial z}\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
\end{align}
```

Writing out the final result fully for ions

```math
\begin{align}
  & \frac{\partial g_{i}}{\partial t}
  + \frac{v_{\mathrm{th},i}}{n_{i}}\left(v_{\mathrm{th},i}w_{\|,i}
  + u_{i}\right)\frac{\partial f_{i}}{\partial z}
  + \left(\frac{3}{2}R_{\mathrm{ion}}n_{n}
  - \frac{u_{i}}{n_{i}}\frac{\partial n_{i}}{\partial z}\right)g_{i} \\
  & + \left(\frac{u_{i}}{v_{\mathrm{th},i}}\frac{\partial v_{\mathrm{th},i}}{\partial z}
  + \frac{1}{2p_{\|,i}}\frac{\partial q_{\|,i}}{\partial z}\right. \\
  & \qquad + \frac{1}{2p_{\|,i}}R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n}
  - n_{i}n_{n}\left(u_{i} - u_{n}\right)^{2}\right) \\
  & \qquad \left. - \frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,i}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{i}\right)^{2}\right)\right)g_{i} \\
  & + \left[-\frac{1}{n_{i}v_{\mathrm{th},i}}\left(-\frac{\partial p_{\|,i}}{\partial z}
  + R_{in}n_{i}n_{n}\left(u_{n} - u_{i}\right)
  + R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{i}\right)\right)\right. \\
  & \qquad-\frac{w_{\|,i}}{2}\frac{1}{p_{\|,i}}\left(-\frac{\partial q_{\|,i}}{\partial z}
  - R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n}
  - n_{i}n_{n}\left(u_{i} - u_{n}\right)^{2}\right)
  + v_{\mathrm{th},i}w_{\|,i}\frac{\partial p_{\|,i}}{\partial z}\right) \\
  & \qquad - \frac{w_{\|,i}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,i}}
  - \frac{n_{n}}{n_{i}} + \frac{n_{n}}{p_{\|,i}}\left(u_{n} - u_{i}\right)^{2}\right) \\
  & \qquad\left. + \frac{w_{\|,i}^{2}}{2}\frac{v_{\mathrm{th},i}}{n_{i}}\frac{\partial n_{i}}{\partial z}\right]\frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = -R_{in}n_{n}\left(g_{i} - \frac{v_{\mathrm{th},i}}{v_{\mathrm{th},n}}g_{n}\right)
      + R_{\mathrm{ion}}v_{\mathrm{th},i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
\end{align}
```

and for neutrals where several of the ionization terms cancel

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{n}}{\partial t}
  + \frac{v_{\mathrm{th},n}}{n_{n}}\left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\frac{\partial f_{n}}{\partial z}
  + \left(-\frac{3}{2}R_{\mathrm{ion}}n_{i}
  - \frac{u_{n}}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)g_{n} \\
  & + \left(\frac{u_{n}}{v_{\mathrm{th},n}}\frac{\partial v_{\mathrm{th},n}}{\partial z}
  + \frac{1}{2p_{\|,n}}\frac{\partial q_{\|,n}}{\partial z}\right. \\
  & \qquad \left. + \frac{1}{2p_{\|,n}}R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i}
  - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right)
  + \frac{1}{2}R_{\mathrm{ion}}n_{i}\right)g_{n} \\
  & + \left[-\frac{1}{n_{n}v_{\mathrm{th},n}}\left(-\frac{\partial p_{\|,n}}{\partial z}
  + R_{in}n_{n}n_{i}\left(u_{i} - u_{n}\right)\right)\right. \\
  & \qquad-\frac{w_{\|,n}}{2}\frac{1}{p_{\|,n}}\left(-\frac{\partial q_{\|,n}}{\partial z}
  - R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i}
  - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right)
  + v_{\mathrm{th},n}w_{\|,n}\frac{\partial p_{\|,n}}{\partial z}\right) \\
  & \qquad\left. + \frac{w_{\|,n}^{2}}{2}\frac{v_{\mathrm{th},n}}{n_{n}}\frac{\partial n_{n}}{\partial z}\right]\frac{\partial g_{n}}{\partial w_{\|,n}} \\
  & = -R_{in}n_{i}\left(g_{n} - \frac{v_{\mathrm{th},n}}{v_{\mathrm{th},i}}g_{i}\right)
      - R_{\mathrm{ion}}n_{i}g_{n}
\end{align}
```

#### Implemented

The kinetic equation as implemented in the code is given in its different variants below.

##### Drift kinetic (full-f)

```math
\begin{align}
  \frac{\partial f_{i}}{\partial t} & =
      - \dot{v}_{\|,i} \frac{\partial f_{i}}{\partial v_{\|}}
      - \dot{z}_{i} \frac{\partial f_{i}}{\partial z}
      + C_{\mathrm{CX},in}
      + C_{\mathrm{ion},i}
  \\
  \dot{v}_{\|,i} & = -\frac{1}{2}\frac{\partial \phi}{\partial z} \\
  \dot{z}_{i} & = v_{\|} \\
  C_{\mathrm{CX},in} & = R_{in} \left( f_{n} n_{i} - f_{i} n_{n} \right) \\
  C_{\mathrm{ion},i} & = R_{\mathrm{ion}} f_{n} n_{i}
\end{align}
```

 $\dot{v}_{\|,i}$ implemented in [`vpa_advection.update_speed_default!`](@ref
moment_kinetics.vpa_advection.update_speed_default!).

 $\dot{z}_{i}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!).

 $C_{\mathrm{CX},in}$ implemented in
 [`charge_exchange.charge_exchange_collisions!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions!).

 $C_{\mathrm{ion},i}$ implemented in
 [`ionization.ionization_collisions!`](@ref
 moment_kinetics.ionization.ionization_collisions!).

```math
\begin{align}
  \frac{\partial f_{n}}{\partial t} & =
      - \dot{z}_{n} \frac{\partial f_{n}}{\partial z}
      + C_{\mathrm{CX},ni}
      + C_{\mathrm{ion},n}
  \\
  \dot{z}_{n} & = v_{\|} \\
  C_{\mathrm{CX},in} & = R_{in} \left( f_{i} n_{n} - f_{n} n_{i} \right) \\
  C_{\mathrm{ion},n} & = -R_{\mathrm{ion}} f_{n} n_{i}
\end{align}
```

 $\dot{z}_{n}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!).

 $C_{\mathrm{CX},ni}$ implemented in
 [`charge_exchange.charge_exchange_collisions!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions!).

 $C_{\mathrm{ion},n}$ implemented in
 [`ionization.ionization_collisions!`](@ref
 moment_kinetics.ionization.ionization_collisions!).

##### Evolving density

With the normalised distribution function
```math
\begin{align}
  g_{s}(z,v_{\|}) = \frac{f_{s}(z,v_{\|})}{n_{s}}
\end{align}
```

```math
\begin{align}
  \frac{\partial g_{i}}{\partial t} & =
      - \dot{v}_{\|,i} \frac{\partial g_{i}}{\partial v_{\|}}
      - \dot{z}_{i} \frac{\partial (n_{i} g_{i})}{\partial z}
      - \dot{g}_{i}
      + C_{\mathrm{CX},in}
      + C_{\mathrm{ion},i}
  \\
  \dot{v}_{\|,i} & = -\frac{1}{2}\frac{\partial \phi}{\partial z} \\
  \dot{z}_{i} & = \frac{v_{\|}}{n_{i}} \\
  \dot{g}_{i} & = - \frac{1}{n_{i}} \frac{\partial (n_{i} u_{i})}{\partial z} g_{i} \\
  C_{\mathrm{CX},in} & = R_{in} n_{n} \left( g_{n} - g_{i} \right) \\
  C_{\mathrm{ion},i} & = R_{\mathrm{ion}} n_{n} \left( g_{n} - g_{i} \right)
\end{align}
```

 $\dot{v}_{\|,i}$ implemented in [`vpa_advection.update_speed_default!`](@ref
moment_kinetics.vpa_advection.update_speed_default!).

 $\dot{z}_{i}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{i}$ implemented in
 [`source_terms.source_terms_evolve_density!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_density!).

 $C_{\mathrm{CX},in}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},i}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).

```math
\begin{align}
  \frac{\partial g_{n}}{\partial t} & =
      - \dot{z}_{n} \frac{\partial (n_{n} g_{n})}{\partial z}
      - \dot{g}_{n}
      + C_{\mathrm{CX},ni}
  \\
  \dot{z}_{n} & = \frac{v_{\|}}{n_{n}} \\
  \dot{g}_{n} & = - \frac{1}{n_{n}} \frac{\partial (n_{n} u_{n})}{\partial z} g_{n} \\
  C_{\mathrm{CX},ni} & = R_{in} n_{i} \left( g_{i} - g_{n} \right)
\end{align}
```

 $\dot{z}_{n}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{n}$ implemented in
 [`source_terms.source_terms_evolve_density!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_density!).

 $C_{\mathrm{CX},ni}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},n}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).

##### Evolving density and parallel flow

With the normalised velocity coordinate
```math
\begin{align}
  w_{\|,s} & = v_{\|} - u_{s}
\end{align}
```
and the normalised distribution function
```math
\begin{align}
  g_{s}(z,w_{\|,s}) = \frac{f_{s}(z,v_{\|}(w_{\|,s}))}{n_{s}}
\end{align}
```

```math
\begin{align}
  \frac{\partial g_{i}}{\partial t} & =
      - \dot{w}_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}}
      - \dot{z}_{i} \frac{\partial (n_{i} g_{i})}{\partial z}
      - \dot{g}_{i}
      + C_{\mathrm{CX},in}
      + C_{\mathrm{ion},i}
  \\
  \dot{w}_{\|,i} & = \frac{1}{n_{i}} \frac{\partial p_{\|,i}}{\partial z}
                     - w_{\|,i} \frac{\partial u_{i}}{\partial z}
                     - R_{in} n_{n} \left(u_{n} - u_{i}\right)
                     - R_{\mathrm{ion}} n_{n} \left(u_{n} - u_{i}\right)
  \\
  \dot{z}_{i} & = \frac{w_{\|} + u_{i}}{n_{i}} \\
  \dot{g}_{i} & = - \frac{1}{n_{i}} \frac{\partial (n_{i} u_{i})}{\partial z} g_{i} \\
  C_{\mathrm{CX},in} & = R_{in} n_{n} \left( g_{n}(z,w_{\|,n}(w_{\|,i})) - g_{i} \right) \\
  C_{\mathrm{ion},i} & = R_{\mathrm{ion}} n_{n} \left( g_{n}(z,w_{\|,n}(w_{\|,i})) - g_{i} \right)
\end{align}
```

 $\dot{w}_{\|,i}$ implemented in
[`vpa_advection.update_speed_n_u_evolution!`](@ref
moment_kinetics.vpa_advection.update_speed_n_u_evolution!).

 $\dot{z}_{i}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{i}$ implemented in
 [`source_terms.source_terms_evolve_density!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_density!).

 $C_{\mathrm{CX},in}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},i}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).

```math
\begin{align}
  \frac{\partial g_{n}}{\partial t} & =
      - \dot{w}_{\|,n} \frac{\partial g_{n}}{\partial w_{\|,n}}
      - \dot{z}_{n} \frac{\partial (n_{n} g_{n})}{\partial z}
      - \dot{g}_{n}
      + C_{\mathrm{CX},ni}
  \\
  \dot{w}_{\|,n} & = \frac{1}{n_{n}} \frac{\partial p_{\|,n}}{\partial z}
                     - w_{\|,n} \frac{\partial u_{n}}{\partial z}
                     - R_{in} n_{i} \left(u_{i} - u_{n}\right)
  \\
  \dot{z}_{n} & = \frac{w_{\|} + u_{n}}{n_{n}} \\
  \dot{g}_{n} & = - \frac{1}{n_{n}} \frac{\partial (n_{n} u_{n})}{\partial z} g_{n} \\
  C_{\mathrm{CX},ni} & = R_{in} n_{i} \left( g_{i}(z,w_{\|,i}(w_{\|,n})) - g_{n} \right)
\end{align}
```

 $\dot{w}_{\|,n}$ implemented in
[`vpa_advection.update_speed_n_u_evolution!`](@ref
moment_kinetics.vpa_advection.update_speed_n_u_evolution!).

 $\dot{z}_{n}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{n}$ implemented in
 [`source_terms.source_terms_evolve_density!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_density!).

 $C_{\mathrm{CX},ni}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},n}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).

##### Evolving density, parallel flow and parallel pressure

With the normalised velocity coordinate
```math
\begin{align}
  w_{\|,s} & = \frac{v_{\|} - u_{s}}{v_{\mathrm{th},s}} \\
  v_{\mathrm{th},s} & = \sqrt{\frac{2 p_{\|,s}}{n_{s}}}
\end{align}
```
and the normalised distribution function
```math
\begin{align}
  g_{s}(z,w_{\|,s}) = \frac{v_{\mathrm{th},s} f_{s}(z,v_{\|}(w_{\|,s}))}{n_{s}}
\end{align}
```

```math
\begin{align}
  \frac{\partial g_{i}}{\partial t} & =
      - \dot{w}_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}}
      - \dot{z}_{i} \frac{\partial (n_{i} g_{i} / v_{\mathrm{th},i})}{\partial z}
      - \dot{g}_{i}
      + C_{\mathrm{CX},in}
      + C_{\mathrm{ion},i}
  \\
  \dot{w}_{\|,i} & = \frac{1}{n_{i} v_{\mathrm{th},i}} \frac{\partial p_{\|,i}}{\partial z}
                     + \frac{w_{\|,i}}{2 p_{\|,i}}\frac{\partial q_{\|,i}}{\partial z}
                     - w_{\|,i}^2 \frac{\partial v_{\mathrm{th},i}}{\partial z} \\
              &\quad + R_{in} \left(
                                    \frac{w_{\|,i}}{2 p_{\|,i}}
                                    \left(
                                          n_{n} p_{\|,i} - n_{i} p_{\|,n}
                                          - n_{i} n_{n} \left( u_{i} - u_{n} \right)^2
                                    \right)
                                    - \frac{n_{n}}{v_{\mathrm{th},i}} \left( u_{n} - u_{i} \right)
                              \right) \\
              &\quad + R_{\mathrm{ion}} \left(
                                              \frac{w_{\|,i}}{2}
                                              \left(
                                                    n_{n} - n_{i} \frac{p_{\|,n}}{p_{\|,i}}
                                                    - \frac{n_{i} n_{n}}{p_{\|,i}} \left( u_{n} - u_{i} \right)^2
                                              \right)
                                              - \frac{n_{n}}{v_{\mathrm{th},i}} \left( u_{n} - u_{i} \right)
                                        \right)
  \\
  \dot{z}_{i} & = \frac{v_{\mathrm{th},i}}{n_{i}} \left(w_{\|} v_{\mathrm{th},i} + u_{i}\right)
  \\
  \dot{g}_{i} & = - \left(
                          \frac{u_{i}}{n_{i}} \frac{\partial n_{i}}{\partial z}
                          - \frac{u_{i}}{v_{\mathrm{th},i}} \frac{\partial v_{\mathrm{th},i}}{\partial z}
                          - \frac{1}{2 p_{\|,i}} \frac{\partial q_{\|,i}}{\partial z}
                    \right) g_{i} \\
           &\quad + \frac{1}{2}
                    \left(
                          \frac{R_{in}}{p_{\|,i}} \left(
                                       n_{n} p_{\|,i} - n_{i} p_{\|,n}
                                       - n_{i} n_{n} \left(u_{i} - u_{n}\right)^2
                                 \right)
                          + R_{\mathrm{ion}}
                            \left(
                                 3 n_{n}
                                 - \frac{n_{i}}{p_{\|,i}}
                                   \left( p_{\|,n} + n_{n} \left(u_{i} - u_{n}\right) \right)^2
                           \right)
                    \right) g_{i}
  \\
  C_{\mathrm{CX},in} & = R_{in} n_{n}
                         \left(
                               g_{n}(z,w_{\|,n}(w_{\|,i})) \frac{v_{\mathrm{th},i}}{v_{\mathrm{th},n}}
                               - g_{i}
                         \right)
  \\
  C_{\mathrm{ion},i} & = R_{\mathrm{ion}} n_{n}
                         \left(
                               g_{n}(z,w_{\|,n}(w_{\|,i})) \frac{v_{\mathrm{th},i}}{v_{\mathrm{th},n}}
                               - g_{i}
                         \right)
\end{align}
```

 $\dot{w}_{\|,i}$ implemented in
[`vpa_advection.update_speed_n_u_p_evolution!`](@ref
moment_kinetics.vpa_advection.update_speed_n_u_p_evolution!).

 $\dot{z}_{i}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{i}$ implemented in
 [`source_terms.source_terms_evolve_ppar_no_collisions!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_ppar_no_collisions!) and
 [`source_terms.source_terms_evolve_ppar_collisions!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_ppar_collisions!).

 $C_{\mathrm{CX},in}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},i}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).

```math
\begin{align}
  \frac{\partial g_{n}}{\partial t} & =
      - \dot{w}_{\|,n} \frac{\partial g_{n}}{\partial w_{\|,n}}
      - \dot{z}_{n} \frac{\partial (n_{n} g_{n} / v_{\mathrm{th},n})}{\partial z}
      - \dot{g}_{n}
      + C_{\mathrm{CX},ni}
  \\
  \dot{w}_{\|,n} & = \frac{1}{n_{n} v_{\mathrm{th},n}} \frac{\partial p_{\|,n}}{\partial z}
                     + \frac{w_{\|,n}}{2 p_{\|,n}}\frac{\partial q_{\|,n}}{\partial z}
                     - w_{\|,n}^2 \frac{\partial v_{\mathrm{th},n}}{\partial z} \\
              &\quad + R_{in} \left(
                                    \frac{w_{\|,n}}{2 p_{\|,n}}
                                    \left(
                                          n_{i} p_{\|,n} - n_{n} p_{\|,i}
                                          - n_{n} n_{i} \left(u_{n} - u_{i}\right)^2
                                    \right)
                                    - \frac{n_{i}}{v_{\mathrm{th},n}} \left(u_{i} - u_{n}\right)
                              \right)
  \\
  \dot{z}_{n} & = \frac{v_{\mathrm{th},n}}{n_{n}} \left(w_{\|} v_{\mathrm{th},n} + u_{n}\right)
  \\
  \dot{g}_{n} & = - \left(
                          \frac{u_{n}}{n_{n}} \frac{\partial n_{n}}{\partial z}
                          - \frac{u_{n}}{v_{\mathrm{th},n}} \frac{\partial v_{\mathrm{th},n}}{\partial z}
                          - \frac{1}{2 p_{\|,n}} \frac{\partial q_{\|,n}}{\partial z}
                    \right) g_{n} \\
           &\quad + \frac{1}{2}
                    \left(
                          \frac{R_{in}}{p_{\|,n}}
                          \left(
                                n_{i} p_{\|,n} - n_{n} p_{\|,i}
                                - n_{n} n_{i} \left( u_{n} - u_{i} \right)^2
                          \right)
                          - 2 R_{\mathrm{ion}} n_{i}
                    \right) g_{n}
  \\
  C_{\mathrm{CX},ni} & = R_{in} n_{i}
                         \left(
                               g_{i}(z,w_{\|,i}(w_{\|,n})) \frac{v_{\mathrm{th},n}}{v_{\mathrm{th},i}}
                               - g_{n}
                         \right)
\end{align}
```

 $\dot{w}_{\|,n}$ implemented in
[`vpa_advection.update_speed_n_u_p_evolution!`](@ref
moment_kinetics.vpa_advection.update_speed_n_u_p_evolution!).

 $\dot{z}_{n}$ implemented in [`z_advection.update_speed_z!`](@ref
moment_kinetics.z_advection.update_speed_z!) and
[`z_advection.adjust_advection_speed!`](@ref
moment_kinetics.z_advection.adjust_advection_speed!).

 $\dot{g}_{n}$ implemented in
 [`source_terms.source_terms_evolve_ppar_no_collisions!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_ppar_no_collisions!) and
 [`source_terms.source_terms_evolve_ppar_collisions!`](@ref
 moment_kinetics.source_terms.source_terms_evolve_ppar_collisions!).

 $C_{\mathrm{CX},ni}$ implemented in
 [`charge_exchange.charge_exchange_collisions_single_species!`](@ref
 moment_kinetics.charge_exchange.charge_exchange_collisions_single_species!).

 $C_{\mathrm{ion},n}$ implemented in
 [`ionization.ionization_collisions_single_species!`](@ref
 moment_kinetics.ionization.ionization_collisions_single_species!).
