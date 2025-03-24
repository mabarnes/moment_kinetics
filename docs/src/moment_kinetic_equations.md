Moment kinetic equations
========================

The following are partial notes on the derivation of the equations being solved
by moment\_kinetics. It would be useful to expand them with more details from
the Excalibur/Neptune reports. Equation references give the report number and
equation number, e.g. (TN-04;1) is equation (1) from report TN-04.pdf.

The drift kinetic equation (DKE), marginalised over $v_{\perp}$, for ions is,
adding ionization and a source term to the form in (TN-04;1),

```math
\begin{align}
  \frac{\partial f_{i}}{\partial t}
  +v_{\|}\frac{\partial f_{i}}{\partial z}
  -\frac{e}{m}\frac{\partial\phi}{\partial z}\frac{\partial f_{i}}{\partial v_{\|}}
  &= -R_{\mathrm{in}}\left(n_{n}f_{i}-n_{i}f_{n}\right)+R_{\mathrm{ion}}n_{i}f_{n}
    + S_i,
\end{align}
```

and for neutrals, adding ionization and a source term to (TN-04;2)
```math
\begin{align}
  \frac{\partial f_{n}}{\partial t}
  +v_{\|}\frac{\partial f_{n}}{\partial z}
  &= -R_{\mathrm{in}}\left(n_{i}f_{n}-n_{n}f_{i}\right)-R_{\mathrm{ion}}n_{i}f_{n}
    + S_n.
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
  \tilde{S}_i = S_i \frac{c_s\sqrt{\pi}}{N_e} \frac{L_z}{c_s} = S_i \frac{L_z\sqrt{\pi}}{N_e}
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
  &= -\tilde{R}_{in}\left(\tilde{n}_{n}\tilde{f}_{i}-\tilde{n}_{i}\tilde{f}_{n}\right)
    + \tilde{R}_{\mathrm{ion}}\tilde{n}_{i}\tilde{f}_{n}
    + \tilde{S}_i
\end{align}
```

and the neutral DKE is

```math
\begin{align}
  \frac{\partial\tilde{f}_{n}}{\partial\tilde{t}}
  + v_{\|}\frac{\partial\tilde{f}_{n}}{\partial\tilde{z}}
  &= -\tilde{R}_{in}\left(\tilde{n}_{i}\tilde{f}_{n}-\tilde{n}_{n}\tilde{f}_{i}\right)
    - \tilde{R}_{\mathrm{ion}}\tilde{n}_{i}\tilde{f}_{n}
    + \tilde{S}_n.
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
    = \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}^{2}\tilde{f}_{s}
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
    &= \frac{1}{\sqrt{\pi}}\int d\tilde{v}_{\|}\tilde{v}_{\|}^{3}\tilde{f}_{s}
       - 3\tilde{u}_{s}\tilde{p}_{\|,s}
       - \tilde{n}_{s}\tilde{u}_{s}^{3}
\end{align}
```

we can take moments of the ion DKE to give ion moment equations (dropping
tildes from here on)

```math
\begin{align}
  \frac{\partial n_{i}}{\partial t}+\frac{\partial\left(n_{i}u_{i}\right)}{\partial z}
  & = -R_{in}\left(n_{n}n_{i}-n_{i}n_{n}\right)+R_{\mathrm{ion}}n_{i}n_{n}
      + \int dv_\parallel S_i\\
%
  & = R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_i
\end{align}
```

```math
\begin{align}
  \frac{\partial\left(n_{i}u_{i}\right)}{\partial t} + \frac{\partial\left(p_{\|,i}
  + n_{i}u_{i}^{2}\right)}{\partial z} + \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  &= -R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
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
  n_{i}\frac{\partial u_{i}}{\partial t}
  + u_{i}\left(R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{i}\right)
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
  &= -R_{in}n_{n}\left(u_{i}-u_{n}\right)
    + R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n}-u_{i}\right)
    - \frac{u_{i}}{n_{i}} \int dv_\parallel S_{i}
\end{align}
```

```math
\begin{align}
  & \frac{\partial\left(p_{\|,i} + n_{i}u_{i}^{2}\right)}{\partial t}
    + \frac{\partial\left(q_{\|,i} + 3u_{i}p_{\|,i}
    + n_{i}u_{i}^{3}\right)}{\partial z} + \frac{\partial\phi}{\partial z}n_{i}u_{i} \\
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n}+n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{i} \\
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
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{i} \\
%
  \frac{p_{\|,i}}{\partial t} + 2u_{i}\frac{\partial n_{i}u_{i}}{\partial t}
  - u_{i}^{2}\frac{\partial n_{i}}{\partial t} + \frac{\partial\left(q_{\|,i}
  + 3u_{i}p_{\|,i} + n_{i}u_{i}^{3}\right)}{\partial z}
  + \frac{\partial\phi}{\partial z}n_{i}u_{i}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{i} \\
%
  \frac{\partial p_{\|,i}}{\partial t} + 2u_{i}\left(-\frac{\partial p_{\|,i}}{\partial z}
  - \frac{\partial\left(n_{i}u_{i}^{2}\right)}{\partial z}
  - \frac{1}{2}\frac{\partial\phi}{\partial z}n_{i}
  - R_{in}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)
  + R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right) \\
  -u_{i}^{2}\left(-\frac{\partial\left(n_{i}u_{i}\right)}{\partial z}
  + R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{i}\right)
  + \frac{\partial q_{\|,i}}{\partial z}
  + \frac{\partial\left(3u_{i}p_{\|,i}\right)}{\partial z}
  + \frac{\partial\left(n_{i}u_{i}^{3}\right)}{\partial z}
  + \frac{\partial\phi}{\partial z}n_{i}u_{i}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right)
      - n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)\right)
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{i} \\
%
  \frac{\partial p_{\|,i}}{\partial t} + u_{i}\frac{\partial p_{\|,i}}{\partial z}
  + 3p_{\|,i}\frac{\partial u_{i}}{\partial z} + \frac{\partial q_{\|,i}}{\partial z}
  & = -R_{in}\left(n_{n}\left(p_{\|,i} + n_{i}u_{i}^{2}\right) - n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2}\right) - 2u_{i}\left(n_{n}n_{i}u_{i} - n_{i}n_{n}u_{n}\right)\right) \\
      & \quad + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2} + n_{n}u_{i}^{2}
      - 2n_{n}u_{i}u_{n}\right)
      + \int dv_\parallel v_\parallel^2 S_{i} + u_{i}^2 \int dv_\parallel S_{i} \\
%
  \frac{\partial p_{\|,i}}{\partial t} + u_{i}\frac{\partial p_{\|,i}}{\partial z}
  + 3p_{\|,i}\frac{\partial u_{i}}{\partial z} + \frac{\partial q_{\|,i}}{\partial z}
  & = -R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n} - n_{i}n_{n}\left(u_{i}^{2} + u_{n}^{2}
      - 2u_{i}u_{n}\right)\right) + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}\left(u_{n}
      - u_{i}\right)^{2}\right) \\
      & \quad + \int dv_\parallel v_\parallel^2 S_{i} + u_{i}^2 \int dv_\parallel S_{i} \\
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
      + R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}\left(u_{n} - u_{i}\right)^{2}\right) \\
      & \quad + \int dv_\parallel v_\parallel^2 S_{i} + u_{i}^2 \int dv_\parallel S_{i} \\
\end{align}
```

and of the neutral DKE to give neutral moment equations

```math
\begin{align}
  \frac{\partial n_{n}}{\partial t} + \frac{\partial\left(n_{n}u_{n}\right)}{\partial z}
  & = -R_{i}\left(n_{i}n_{n} - n_{n}n_{i}\right) - R_{\mathrm{ion}}n_{i}n_{n}
      + \int dv_\parallel S_{n} \\
%
  & =-R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{n}
\end{align}
```

```math
\begin{align}
  \frac{\partial\left(n_{n}u_{n}\right)}{\partial t}
  + \frac{\partial\left(p_{\|,n} + n_{n}u_{n}^{2}\right)}{\partial z}
  &= -R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
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
  + u_{n}\left(-R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{n}\right)
  + \frac{\partial p_{\|,n}}{\partial z}
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
  &= -R_{in}n_{i}\left(u_{n} - u_{i}\right) - \frac{u_{n}}{n_{n}} \int dv_\parallel S_{n}
\end{align}
```

```math
\begin{align}
  & \frac{\partial\left(p_{\|,n} + n_{n}u_{n}^{2}\right)}{\partial t}
    + \frac{\partial\left(q_{\|,n} + 3u_{n}p_{\|,n}
    + n_{n}u_{n}^{3}\right)}{\partial z} + q_{n}\frac{\partial\phi}{\partial z}n_{n}u_{n} \\
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right)\right)
      - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{n} \\
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
      - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{n} \\
%
  \frac{\partial p_{\|,n}}{\partial t} + 2u_{n}\frac{\partial n_{n}u_{n}}{\partial t}
  - u_{n}^{2}\frac{\partial n_{n}}{\partial t} + \frac{\partial\left(q_{\|,n}
  + 3u_{n}p_{\|,n} + n_{n}u_{n}^{3}\right)}{\partial z}
  + q_{n}\frac{\partial\phi}{\partial z}n_{n}u_{n}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right)\right) - R_{\mathrm{ion}}n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2}\right)
      + \int dv_\parallel v_\parallel^2 S_{n} \\
%
  \frac{\partial p_{\|,n}}{\partial t}
  + 2u_{n}\left(-\frac{\partial p_{\|,n}}{\partial z}
  - \frac{\partial\left(n_{n}u_{n}^{2}\right)}{\partial z}
  - \frac{q_{n}}{2}\frac{\partial\phi}{\partial z}n_{n}
  - R_{in}\left(n_{i}n_{n}u_{n} - n_{n}n_{i}u_{i}\right)
  - R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right) \\
  - u_{n}^{2}\left(-\frac{\partial\left(n_{n}u_{n}\right)}{\partial z}
  - R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{n}\right)
  + \frac{\partial q_{\|,n}}{\partial z}
  + \frac{\partial\left(3u_{n}p_{\|,n}\right)}{\partial z}
  + \frac{\partial\left(n_{n}u_{n}^{3}\right)}{\partial z}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
  + n_{i}u_{i}^{2}\right)\right)
  - R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{n} \\
%
  \frac{\partial p_{\|,n}}{\partial t} + u_{n}\frac{\partial p_{\|,n}}{\partial z}
  + 3p_{\|,n}\frac{\partial u_{n}}{\partial z} + \frac{\partial q_{\|,n}}{\partial z}
  & = -R_{in}\left(n_{i}\left(p_{\|,n} + n_{n}u_{n}^{2}\right) - n_{n}\left(p_{\|,i}
      + n_{i}u_{i}^{2}\right) - 2u_{n}\left(n_{i}n_{n}u_{n}
      - n_{n}n_{i}u_{i}\right)\right) - R_{\mathrm{ion}}n_{i}\left(p_{\|,n}
      + n_{n}u_{n}^{2} + n_{n}u_{n}^{2} - 2n_{n}u_{n}u_{n}\right)
      + \int dv_\parallel v_\parallel^2 S_{n} + u_{n}^2\int dv_\parallel S_{n} \\
%
  \frac{\partial p_{\|,n}}{\partial t} + u_{n}\frac{\partial p_{\|,n}}{\partial z}
  + 3p_{\|,n}\frac{\partial u_{n}}{\partial z} + \frac{\partial q_{\|,n}}{\partial z}
  & = -R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i} - n_{n}n_{i}\left(u_{n}^{2} + u_{i}^{2}
      - 2u_{n}u_{i}\right)\right) - R_{\mathrm{ion}}n_{i}p_{\|,n}
      + \int dv_\parallel v_\parallel^2 S_{n} + u_{n}^2\int dv_\parallel S_{n} \\
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
      - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right) - R_{\mathrm{ion}}n_{i}p_{\|,n} \\
      & \quad + \int dv_\parallel v_\parallel^2 S_{n} + u_{n}^2\int dv_\parallel S_{n} \\
\end{align}
```

Kinetic equation
----------------

For the moment-kinetic equation for the normalized distribution function

```math
\begin{align}
g_{s}(w_{\|,s}) &= \frac{v_{\mathrm{th},s}}{n_{s}}f_{s}(v_{\|}(w_{\|,s}))
\end{align}
```

we transform to the normalized velocity coordinate

```math
\begin{align}
w_{\|,s} &= \frac{v_{\|} - u_{s}}{v_{\mathrm{th},s}}
\end{align}
```

The derivatives transform as

```math
\begin{align}
  \left.\frac{\partial f_{s}}{\partial t}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial t}\right|_{z,w\|}
               - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,t}
               - \frac{w_{\|,s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial t}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,t}\\
%
  \left.\frac{\partial f_{s}}{\partial z}\right|_{z,v\|}
  & \rightarrow\left.\frac{\partial f_{s}}{\partial z}\right|_{t,w\|}
               - \frac{1}{v_{\mathrm{th},s}}\frac{\partial u_{s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,t}
               - \frac{w_{\|,s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial z}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,wt}\\
%
  \left.\frac{\partial f_{s}}{\partial v_{\|}}\right|_{z,v\|}
  & \rightarrow\frac{1}{v_{\mathrm{th},s}}\left.\frac{\partial f_{s}}{\partial w_{\|,s}}\right|_{z,t}
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
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} + S_{i}
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
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} + S_{i} \\
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
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} + S_{i} \\
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
  & = -R_{in}\left(n_{n}f_{i} - n_{i}f_{n}\right) + R_{\mathrm{ion}}n_{i}f_{n} + S_{i}
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
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} + S_{n} \\
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
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} + S_{n} \\
%
  \frac{\partial f_{n}}{\partial t} + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\frac{\partial f_{n}}{\partial z}
  + \left[-\frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial t}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial t}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial t}\right) + \left(v_{\mathrm{th},n}w_{\|,n}
  + u_{n}\right)\left(-\frac{1}{v_{\mathrm{th},n}}\frac{\partial u_{n}}{\partial z}
  - \frac{w_{\|,n}}{2}\left(\frac{1}{p_{\|,n}}\frac{\partial p_{\|,n}}{\partial z}
  - \frac{1}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)\right)\right]\frac{\partial f_{n}}{\partial w_{\|,n}}
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} + S_{n} \\
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
  & = -R_{in}\left(n_{i}f_{n} - n_{n}f_{i}\right) - R_{\mathrm{ion}}n_{i}f_{n} + S_{n}
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
  \frac{\partial f_{s}}{\partial z}
  & = \frac{n_{s}}{v_{\mathrm{th},s}}\frac{\partial g_{s}}{\partial z}
      + \frac{3g_{s}}{2v_{\mathrm{th},s}}\frac{\partial n_{s}}{\partial z}
      - \frac{g_{s}n_{s}}{2v_{\mathrm{th},s}p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z} \\
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
  - \frac{g_{s}n_{s}}{2v_{\mathrm{th},s}p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t} \\
  & + \frac{n_s}{v_{\mathrm{th},s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3 g_s}{2 v_{\mathrm{th},s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial n_s}{\partial z}
  - \frac{g_s n_s}{2 v_{\mathrm{th},s} p_{\parallel,s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial p_{\parallel,s}}{\partial z} \\
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
      \pm R_{\mathrm{ion}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + S_{s} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
  + \frac{3 g_s}{2 n_s} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial n_s}{\partial z}
  - \frac{g_s}{2 p_{\parallel,s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial p_{\parallel,s}}{\partial z} \\
  & + \left[-\frac{1}{v_{\mathrm{th},s}}\left(\frac{\partial u_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial u_{s}}{\partial z}
  + \frac{q_{s}}{2}\frac{\partial\phi}{\partial z}\right)
  - \frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(\frac{\partial p_{\|,s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial p_{\|,s}}{\partial z}\right)
  + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\frac{\partial n_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s}
  + u_{s}\right)\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s} \\
%
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
  + \frac{3 g_s}{2 n_s} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial n_s}{\partial z}
  - \frac{g_s}{2 p_{\parallel,s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial p_{\parallel,s}}{\partial z} \\
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
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s} \\
\end{align}
```

```@raw html
</details>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
  + \frac{3 g_s}{2 n_s} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial n_s}{\partial z}
  - \frac{g_s}{2 p_{\parallel,s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial p_{\parallel,s}}{\partial z} \\
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
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s}
\end{align}
```

So then if we use the moment equations we can rewrite the DKE as

```math
\begin{align}
  & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\frac{\partial n_{s}}{\partial t}
  - \frac{g_{s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial t}
  + \frac{3 g_s}{2 n_s} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial n_s}{\partial z}
  - \frac{g_s}{2 p_{\parallel,s}} \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right) \frac{\partial p_{\parallel,s}}{\partial z} \\
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
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s} \\
\end{align}
```

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}
  - \cancel{u_{s}\frac{\partial n_{s}}{\partial z}}
  - n_{s}\frac{\partial u_{s}}{\partial z}
  + \left(v_{\mathrm{th},s}w_{\|,s} + \cancel{u_{s}}\right) \frac{\partial n_s}{\partial z} \right) \\
  & -\frac{g_{s}}{2p_{\|,s}}\left(-\cancel{u_{s}\frac{\partial p_{\|,s}}{\partial z}}
  - \frac{\partial q_{\|,s}}{\partial z}
  - 3p_{\|,s}\frac{\partial u_{s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n} - u_{s}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}
  + \left(v_{\mathrm{th},s}w_{\|,s} + \cancel{u_{s}}\right) \frac{\partial p_{\parallel,s}}{\partial z} \right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\underbrace{\cancel{n_{s}u_{s}\frac{\partial u_{s}}{\partial z}}}_{A}
  - \frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}u_{n}
  + v_{\mathrm{th},s}w_{\|,s}\left(\underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{B}
  + \underbrace{\cancel{u_{s}\frac{\partial n_{s}}{\partial z}}}_{C}\right)\right)\right. \\
  & \quad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}
  - \underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{A}
  + \underbrace{\cancel{v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}}}_{C}\right) \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - \underbrace{\cancel{3p_{\|,s}\frac{\partial u_{s}}{\partial z}}}_{B}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n}
  - u_{s}\right)^{2}\right) + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}
  - \underbrace{\cancel{n_{s}\frac{\partial u_{s}}{\partial z}}}_{B}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s} \\
%
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \frac{3g_{s}}{2n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}
  + v_{\mathrm{th},s} w_{\parallel,s} \frac{\partial n_{s}}{\partial z} - n_{s}\frac{\partial u_{s}}{\partial z}\right) \\
  & -\frac{g_{s}}{2p_{\|,s}}\left(v_{\mathrm{th},s} w_{\parallel,s}\frac{\partial p_{\|,s}}{\partial z}
  - \frac{\partial q_{\|,s}}{\partial z} - 3p_{\|,s}\frac{\partial u_{s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n} - u_{s}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}\right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)\pm R_{\mathrm{ion}}n_{i}n_{n}u_{n}\right)\right. \\
  & \quad + \frac{u_{s}}{n_{s}v_{\mathrm{th},s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}\right) \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - m_{s}n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \pm R_{\mathrm{ion}}n_{i}\left(p_{\|,n} + m_{s}n_{n}\left(u_{n}
  - u_{s}\right)^{2}\right) + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\pm R_{\mathrm{ion}}n_{i}n_{n} + \int dv_\parallel S_{s}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n}
      + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s}\\
%
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + g_{s}\left(\pm\frac{3}{2}R_{\mathrm{ion}}n_{i}\frac{n_{n}}{n_{s}}
  + \frac{3}{2n_{s}}\int dv_\parallel S_{s}
  + \frac{3 v_{\mathrm{th},s} w_{\parallel,s}}{2n_{s}}\frac{\partial n_{s}}{\partial z}\right) \\
  & + g_{s}\left(-\frac{v_{\mathrm{th},s} w_{\parallel,s}}{2p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z}
  + \frac{1}{2p_{\|,s}}\frac{\partial q_{\|,s}}{\partial z}
  + \frac{1}{2p_{\|,s}}R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  \mp\frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,s}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{s}\right)^{2}\right)
  - \frac{1}{2p_{\parallel,s}}\int dv_\parallel v_\parallel^2 S_{s} - \frac{u_{s}^2}{2p_{\parallel,s}}\int dv_\parallel S_{s}\right) \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{s}\right) - u_{s}\int dv_\parallel S_{s}\right)\right. \\
  & \quad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \quad\mp\frac{w_{\|,s}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,s}}
  - \frac{n_{n}}{n_{s}} + \frac{n_{n}}{p_{\|,s}}\left(u_{n} - u_{s}\right)^{2}\right) \\
  & \quad\left. + \frac{w_{\|,s}}{2}\frac{1}{n_{s}}\left(\int dv_\parallel S_{s} + v_{\mathrm{th},s}w_{\|,s}\frac{\partial n_{s}}{\partial z}\right)\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s}
\end{align*}
```

and finally using

```math
\begin{align*}
  -w_{\parallel,s}\frac{\partial v_{\mathrm{th},s}}{\partial z}
  & =-\frac{v_{\mathrm{th},s} w_{\parallel,s}}{v_{\mathrm{th},s}}\frac{\partial v_{\mathrm{th},s}}{\partial z} \\
  & =-v_{\mathrm{th},s} w_{\parallel,s}\sqrt{\frac{n_{s}}{p_{\|,s}}}\frac{\partial}{\partial z}\sqrt{\frac{p_{\|,s}}{n_{s}}} \\
  & = \frac{v_{\mathrm{th},s} w_{\parallel,s}}{2}\left(-\frac{1}{p_{\|,s}}\frac{\partial p_{\|,s}}{\partial z}
      + \frac{1}{n_{s}}\frac{\partial n_{s}}{\partial z}\right)
\end{align*}
```

gives

```@raw html
</details>
```

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{s}}{\partial t}
  + \left(v_{\mathrm{th},s}w_{\|,s} + u_{s}\right)\frac{\partial g_{s}}{\partial z}
  + \left(\pm\frac{3}{2}R_{\mathrm{ion}}n_{i}\frac{n_{n}}{n_{s}}
  + \frac{3}{2n_{s}} \int dv_\parallel S_{s}
  + \frac{v_{\mathrm{th},s} w_{\parallel,s}}{n_{s}}\frac{\partial n_{s}}{\partial z}\right)g_{s} \\
  & + \left(-w_{\parallel,s}\frac{\partial v_{\mathrm{th},s}}{\partial z}
  + \frac{1}{2p_{\|,s}}\frac{\partial q_{\|,s}}{\partial z}\right. \\
  & \qquad + \frac{1}{2p_{\|,s}}R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right) \\
  & \qquad \left.\mp\frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,s}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{s}\right)^{2}\right)
  - \frac{1}{2p_{\parallel,s}}\int dv_\parallel v_\parallel^2 S_{s} - \frac{u_{s}^2}{2p_{\parallel,s}}\int dv_\parallel S_{s}\right)g_{s} \\
  & + \left[-\frac{1}{n_{s}v_{\mathrm{th},s}}\left(-\frac{\partial p_{\|,s}}{\partial z}
  + R_{ss'}n_{s}n_{s'}\left(u_{s'} - u_{s}\right)
  \pm R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{s}\right) - u_{s}\int dv_\parallel S_{s}\right)\right. \\
  & \qquad-\frac{w_{\|,s}}{2}\frac{1}{p_{\|,s}}\left(-\frac{\partial q_{\|,s}}{\partial z}
  - R_{ss'}\left(n_{s'}p_{\|,s} - n_{s}p_{\|,s'}
  - n_{s}n_{s'}\left(u_{s} - u_{s'}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{s} + u_{s}^2 \int dv_\parallel S_{s}
  + v_{\mathrm{th},s}w_{\|,s}\frac{\partial p_{\|,s}}{\partial z}\right) \\
  & \qquad\mp\frac{w_{\|,s}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,s}}
  - \frac{n_{n}}{n_{s}} + \frac{n_{n}}{p_{\|,s}}\left(u_{n} - u_{s}\right)^{2}\right) \\
  & \qquad\left. + \frac{w_{\parallel,s}}{2}\frac{1}{n_{s}}\int dv_\parallel S_{s}
  + \frac{w_{\|,s}^{2}}{2}\frac{v_{\mathrm{th},s}}{n_{s}}\frac{\partial n_{s}}{\partial z}\right]\frac{\partial g_{s}}{\partial w_{\|,s}} \\
  & = -R_{ss'}n_{s'}\left(g_{s} - \frac{v_{\mathrm{th},s}}{v_{\mathrm{th},s'}}g_{s'}\right)
      \pm R_{\mathrm{ion}}\frac{v_{\mathrm{th},s}}{n_{s}}n_{i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} + \frac{v_{\mathrm{th},s}}{n_{s}} S_{s}
\end{align}
```

Writing out the final result fully for ions

```math
\begin{align}
  & \frac{\partial g_{i}}{\partial t}
  + \left(v_{\mathrm{th},i}w_{\|,i} + u_{i}\right)\frac{\partial g_{i}}{\partial z}
  + \left(\frac{3}{2}R_{\mathrm{ion}}n_{n} + \frac{3}{2n_{i}}\int dv_\parallel S_{i}
  + \frac{v_{\mathrm{th},i} w_{\parallel,i}}{n_{i}}\frac{\partial n_{i}}{\partial z}\right)g_{i} \\
  & + \left(-w_{\parallel,i}\frac{\partial v_{\mathrm{th},i}}{\partial z}
  + \frac{1}{2p_{\|,i}}\frac{\partial q_{\|,i}}{\partial z}\right. \\
  & \qquad + \frac{1}{2p_{\|,i}}R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n}
  - n_{i}n_{n}\left(u_{i} - u_{n}\right)^{2}\right) \\
  & \qquad \left. - \frac{1}{2}R_{\mathrm{ion}}\frac{n_{i}}{p_{\|,i}}\left(p_{\|,n}
  + n_{n}\left(u_{n} - u_{i}\right)^{2}\right)
  - \frac{1}{2p_{\parallel,i}}\int dv_\parallel v_\parallel^2 S_{i} - \frac{u_{i}^2}{2p_{\parallel,i}}\int dv_\parallel S_{i}\right)g_{i} \\
  & + \left[-\frac{1}{n_{i}v_{\mathrm{th},i}}\left(-\frac{\partial p_{\|,i}}{\partial z}
  + R_{in}n_{i}n_{n}\left(u_{n} - u_{i}\right)
  + R_{\mathrm{ion}}n_{i}n_{n}\left(u_{n} - u_{i}\right) - u_{i}\int dv_\parallel S_{i}\right)\right. \\
  & \qquad-\frac{w_{\|,i}}{2}\frac{1}{p_{\|,i}}\left(-\frac{\partial q_{\|,i}}{\partial z}
  - R_{in}\left(n_{n}p_{\|,i} - n_{i}p_{\|,n}
  - n_{i}n_{n}\left(u_{i} - u_{n}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{i} + u_{i}^2 \int dv_\parallel S_{i}
  + v_{\mathrm{th},i}w_{\|,i}\frac{\partial p_{\|,i}}{\partial z}\right) \\
  & \qquad - \frac{w_{\|,i}}{2}R_{\mathrm{ion}}n_{i}\left(\frac{p_{\|,n}}{p_{\|,i}}
  - \frac{n_{n}}{n_{i}} + \frac{n_{n}}{p_{\|,i}}\left(u_{n} - u_{i}\right)^{2}\right) \\
  & \qquad\left. + \frac{w_{\parallel,i}}{2} \frac{1}{n_{i}}\int dv_\parallel S_{i}
  + \frac{w_{\|,i}^{2}}{2}\frac{v_{\mathrm{th},i}}{n_{i}}\frac{\partial n_{i}}{\partial z}\right]\frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = -R_{in}n_{n}\left(g_{i} - \frac{v_{\mathrm{th},i}}{v_{\mathrm{th},n}}g_{n}\right)
      + R_{\mathrm{ion}}v_{\mathrm{th},i}\frac{n_{n}}{v_{\mathrm{th},n}}g_{n} + \frac{v_{\mathrm{th},i}}{n_{i}} S_{i}
\end{align}
```

and for neutrals where several of the ionization terms cancel

```math
\begin{align}
  \Rightarrow & \frac{\partial g_{n}}{\partial t}
  + \left(v_{\mathrm{th},n}w_{\|,n} + u_{n}\right)\frac{\partial f_{n}}{\partial z}
  + \left(-\frac{3}{2}R_{\mathrm{ion}}n_{i} + \frac{3}{2n_{n}}\int dv_\parallel S_{n}
  + \frac{v_{\mathrm{th},n} w_{\parallel,n}}{n_{n}}\frac{\partial n_{n}}{\partial z}\right)g_{n} \\
  & + \left(-w_{\parallel,n}\frac{\partial v_{\mathrm{th},n}}{\partial z}
  + \frac{1}{2p_{\|,n}}\frac{\partial q_{\|,n}}{\partial z}\right. \\
  & \qquad \left. + \frac{1}{2p_{\|,n}}R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i}
  - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right)
  + \frac{1}{2}R_{\mathrm{ion}}n_{i}
  - \frac{1}{2p_{\parallel,n}}\int dv_\parallel v_\parallel^2 S_{n} - \frac{u_{n}^2}{2p_{\parallel,n}}\int dv_\parallel S_{n}\right)g_{n} \\
  & + \left[-\frac{1}{n_{n}v_{\mathrm{th},n}}\left(-\frac{\partial p_{\|,n}}{\partial z}
  + R_{in}n_{n}n_{i}\left(u_{i} - u_{n}\right) - u_{n}\int dv_\parallel S_{n}\right)\right. \\
  & \qquad-\frac{w_{\|,n}}{2}\frac{1}{p_{\|,n}}\left(-\frac{\partial q_{\|,n}}{\partial z}
  - R_{in}\left(n_{i}p_{\|,n} - n_{n}p_{\|,i}
  - n_{n}n_{i}\left(u_{n} - u_{i}\right)^{2}\right)
  + \int dv_\parallel v_\parallel^2 S_{n} + u_{n}^2\int dv_\parallel S_{n}
  + v_{\mathrm{th},n}w_{\|,n}\frac{\partial p_{\|,n}}{\partial z}\right) \\
  & \qquad\left. + \frac{w_{\parallel,n}}{2}\frac{1}{n_{n}}\int dv_\parallel S_{n}
  + \frac{w_{\|,n}^{2}}{2}\frac{v_{\mathrm{th},n}}{n_{n}}\frac{\partial n_{n}}{\partial z}\right]\frac{\partial g_{n}}{\partial w_{\|,n}} \\
  & = -R_{in}n_{i}\left(g_{n} - \frac{v_{\mathrm{th},n}}{v_{\mathrm{th},i}}g_{i}\right)
      - R_{\mathrm{ion}}n_{i}g_{n} + \frac{v_{\mathrm{th},n}}{n_{n}} S_{n}
\end{align}
```
