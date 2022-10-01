Energy and particle sources
===========================

Particle sources
----------------

No particle sources are currently included. The simulations rely on 100% recycling of ions
to neutrals at the walls to stop the density falling.

Energy sources
--------------

### Ions

An energy source can be added to heat the ions. Its shape (e.g. Gaussian in $z$) and
amplitude can be controlled by input parameters.

Want to define a source that increases $p_{\|}$ but does not change density $n_{s}$ or the
normalised distribution function $g_{s}$.

```math
\begin{equation}
  \frac{\partial f_s}{\partial t} = \ldots + S_\mathrm{heat}(z) F(v_{\|})
\end{equation}
```

where $F(v_{\|})$ will satisfy

```math
\begin{align}
  \int dv_{\|} F(v_{\|}) &= 0 \\
  \int dv_{\|} v_{\|} F(v_{\|}) &= 0
\end{align}
```

so that it does not add particles or parallel momentum. Therefore no source term appears
in the continuity or force balance equations. The second moment is

```math
\begin{equation}
  \int dv_{\|} v_{\|}^{2} F(v_{\|}) = 1
\end{equation}
```

so that in the energy equation

```math
\begin{align}
  \frac{\partial p_{\|}}{\partial t}
  &= \ldots + \int dv_{\|} v_{\|}^{2} S_\mathrm{heat}(z) F(v_{\|})
  &= \ldots + S_\mathrm{heat}(z)
\end{align}
```

In the kinetic equation for $g_{i}$, source terms will appear from the RHS of the ion DKE
and from places where $\partial p_{\|}/\partial t$ appears, see [Kinetic equation](@ref).
So the source contributions would be

```math
\begin{equation}
  \frac{\partial g_{i}}{\partial t}
  = \ldots
     + \frac{v_{\mathrm{th},i}}{n_{i}} S_\mathrm{heat}(z) F(v_{\|})
     + \left(\frac{g_{i}}{2p_{\|,i}}
             + \frac{w_{\|,i}}{2p_{\|,i}}\frac{\partial g_{i}}{\partial w_{\|,i}}\right)
       S_\mathrm{heat}(z)
\end{equation}
```

We want these source contributions to vanish, so 

```math
\begin{equation}
  \frac{v_{\mathrm{th},i}}{n_{i}} S_\mathrm{heat}(z) F(v_{\|})
  + \left(\frac{g_{i}}{2p_{\|,i}}
          + \frac{w_{\|,i}}{2p_{\|,i}}\frac{\partial g_{i}}{\partial w_{\|,i}}\right)
  S_\mathrm{heat}(z)
  = 0
\end{equation}
```

```math
\begin{align}
  F(v_\|)
  &= - \frac{n_{i}}{v_{\mathrm{th},i}}\left(\frac{g_{i}}{2p_{\|,i}}
     + \frac{w_{\|,i}}{2p_{\|,i}}\frac{\partial g_{i}}{\partial w_{\|,i}}\right) \\
  &= - \frac{1}{v_{\mathrm{th},i}^{3}}\left(g_{i}
     + w_{\|,i}\frac{\partial g_{i}}{\partial w_{\|,i}}\right)
\end{align}
```

```@raw html
<details>
<summary>We can check the moments for consistency.</summary>
```

```math
\begin{align*}
  \int dv_{\|} g_{i}
  & = v_{\mathrm{th},i} \int dw_{\|,i} g_{i}
    = v_{\mathrm{th},i} \\

  \int dv_{\|} v_{\|} g_{i}
  & = v_{\mathrm{th},i}^{2} \int dw_{\|,i} \left(w_{\|,i}
      + \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right)g_{i}
    = 0 + v_{\mathrm{th},i}^{2} \frac{u_{\|,i}}{v_{\mathrm{th},i}}
    = v_{\mathrm{th},i} u_{\|,i} \\

  \int dv_{\|} v_{\|}^{2} g_{i}
  & = v_{\mathrm{th},i}^{3} \int dw_{\|,i} \left(w_{\|,i}
      + \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right)^{2}g_{i} \\
  & = v_{\mathrm{th},i}^{3} \int dw_{\|,i} \left(w_{\|,i}^{2}
      + 2w_{\|,i}\frac{u_{\|,i}}{v_{\mathrm{th},i}}
      + \frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}}\right)g_{i} \\
  & = v_{\mathrm{th},i}^{3}
      \left(\frac{1}{2} + 0 + \frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}}\right) \\
  & = v_{\mathrm{th},i}
      \left(\frac{v_{\mathrm{th},i}^{2}}{2} + u_{\|,i}^{2}\right) \\

  \int dv_{\|} w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}}
  & = v_{\mathrm{th},i}
      \int dw_{\|,i} w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = -v_{\mathrm{th},i}
      \int dw_{\|,i} g_{i} \\
  & = -v_{\mathrm{th},i} \\

  \int dv_{\|} v_{\|} w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}}
  & = v_{\mathrm{th},i}^{2}
      \int dw_{\|,i} \left(w_{\|,i} + \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right)
      w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = v_{\mathrm{th},i}^{2}
      \left(-2\int w_{\|,i} w_{\|,i} g_{i}
      - \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right) \\
  & = -v_{\mathrm{th},i}^{2}
      \left(0 + \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right) \\
  & = -v_{\mathrm{th},i}u_{\|,i} \\

  \int dv_{\|} v_{\|}^2 w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}}
  & = v_{\mathrm{th},i}^{3}
      \int dw_{\|,i} \left(w_{\|,i} + \frac{u_{\|,i}}{v_{\mathrm{th},i}}\right)^{2}
      w_{\|,i} \frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = v_{\mathrm{th},i}^{3}
      \int dw_{\|,i} \left(w_{\|,i}^{3}
      + 2\frac{u_{\|,i}}{v_{\mathrm{th},i}}w_{\|,i}^{2}
      + \frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}}w_{\|,i} \right)
      \frac{\partial g_{i}}{\partial w_{\|,i}} \\
  & = v_{\mathrm{th},i}^{3}
      \left(-3\int dw_{\|,i} w_{\|,i}^{2} g_{i}
      - 4\int dw_{\|,i}\frac{u_{\|,i}}{v_{\mathrm{th},i}}w_{\|,i} g_{i}
      - \int dw_{\|,i}\frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}} g_{i} \right) \\
  & = -v_{\mathrm{th},i}^{3}
      \left(\frac{3}{2} + \frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}} \right) \\

  \int dv_{\|}F(v_{\|})
  & = -\frac{1}{v_{\mathrm{th},i}^{3}}
      \left(v_{\mathrm{th},i} - v_{\mathrm{th},i}\right) \\
  & = 0 \\

  \int dv_{\|}v_{\|}F(v_{\|})
  & = -\frac{1}{v_{\mathrm{th},i}^{3}}
      \left(v_{\mathrm{th},i}u_{\|,i} - v_{\mathrm{th},i}u_{\|,i}\right) \\
  & = 0 \\

  \int dv_{\|}v_{\|}^{2}F(v_{\|})
  & = -\frac{1}{v_{\mathrm{th},i}^{3}}
      \left(v_{\mathrm{th},i}\left(\frac{v_{\mathrm{th},i}^{2}}{2}
            + u_{\|,i}^{2}\right)
            - v_{\mathrm{th},i}^{3} \left(\frac{3}{2}
            + \frac{u_{\|,i}^{2}}{v_{\mathrm{th},i}^{2}} \right)\right) \\
  & = -\frac{1}{v_{\mathrm{th},i}^{3}}
      \left(- v_{\mathrm{th},i}^{3}\right) \\
  & = 1
\end{align*}
```

```@raw html
</details>
```

Now write $F(v_{\|})$ in terms of $f_{i}$ and $v_{\|}$ so that we can use it in the full-f
version, using the expression for $\partial g_{i}/\partial w_{\|,i}$ from [Kinetic
equation](@ref).

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align*}
  F(v_\|)
  &= - \frac{1}{v_{\mathrm{th},i}^{3}}\left(\frac{v_{\mathrm{th},i}}{n_{i}}f_{i}
     + w_{\|,i}\frac{v_{\mathrm{th},i}}{n_{i}}\frac{\partial f_{i}}{\partial w_{\|,i}}\right) \\
  &= - \frac{1}{n_{i}v_{\mathrm{th},i}^{2}}\left(f_{i}
     + w_{\|,i}\frac{\partial f_{i}}{\partial w_{\|,i}}\right) \\
  &= - \frac{1}{n_{i}v_{\mathrm{th},i}^{2}}\left(f_{i}
     + w_{\|,i}v_{\mathrm{th},i}\frac{\partial f_{i}}{\partial v_{\|,i}}\right)
\end{align*}
```

```@raw html
</details>
```

```math
\begin{equation}
  F(v_\|)
  = - \frac{1}{n_{i}v_{\mathrm{th},i}^{2}}\left(f_{i}
    + \left(v_\| - u_{\|,i}\right)\frac{\partial f_{i}}{\partial v_{\|,i}}\right)
\end{equation}
```

### Neutrals

Neutrals are emitted with an average energy set by the `T_wall` parameter of the Knudsen
cosine distribution used for the outgoing neutral particles. They may then gain (or lose)
energy by charge exchange interactions with the ions.
