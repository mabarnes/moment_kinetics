Wall boundary conditions with moment constraints
================================================

Ions
----

### Boundary conditions

The sheath-edge boundary conditions for the ions is that no ions leave from the
sheath edge. So at the lower boundary $z=-L_z/2$

```math
\begin{align}
  f(z=-L/2,v_\parallel>0) = 0
\end{align}
```

and at the upper boundary $z=L_z/2$

```math
\begin{align}
  f(z=L/2,v_\parallel<0) = 0
\end{align}
```

### Moment constraints

At the sheath-entrance boundary, the constraints need to be enforced slightly
differently to how they are done in the bulk of the domain (see [Constraints on
normalized distribution function](@ref)). For compatibility with the boundary
condition, the corrections which are added to impose the constraints should go
to zero at $v_\parallel=0$. Note that the constraints are imposed after the
boundary condition is applied by setting $f(v_\parallel>0)=0$ on the lower
sheath boundary or $f(v_\parallel<0)=0$ on the upper sheath boundary.

The form of the correction that we choose is

```math
\begin{align}
\tilde{g}_s &= A\hat{g}_s + Bw_\parallel \frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_s + Cw_\parallel^2 \frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_s
\end{align}
```

We have the same set of constraints

```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dw_{\|}\tilde{g}_{s} & =1\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}\tilde{g}_{s} & =0\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{2}\tilde{g}_{s} & =\frac{1}{2}
\end{align}
```

Defining the integrals

```math
\begin{align}
  I_{n}=\frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{n}\hat{g}_{s}.
  J_{n}=\frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{n}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}.
\end{align}
```

We can write the constraints as

```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dw_{\|}\tilde{g}_{s}=1 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(A\hat{g}_{s}+Bw_{\|}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}+Cw_{\|}^{2}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}\right) \\
  &=AI_{0}+BJ_{1}+CJ_{2}\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}\tilde{g}_{s}=0 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(Aw_{\|}\hat{g}_{s}+Bw_{\|}^{2}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}+Cw_{\|}^{3}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}\right) \\
  &=AI_{1}+BJ_{2}+CJ_{3}\\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{2}\tilde{g}_{s}=\frac{1}{2} & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(Aw_{\|}^{2}\hat{g}_{s}+Bw_{\|}^{3}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}+Cw_{\|}^{4}\frac{|v_\parallel|}{1+|v_\parallel|}\hat{g}_{s}\right) \\
  &=AI_{2}+BJ_{3}+CJ_{4}.
\end{align}
```

and solving these simultaneous equations

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  C &= \frac{\left( \frac{1}{2} - A I_2 - B J_3 \right)}{J_4}  \\
  B &= -\frac{A I_1 + C J_3}{J_2} \\
    &= -\frac{I_1}{J_2} A - \frac{J_3}{J_2} \left( \frac{1}{2J_4} - \frac{I_2}{J_4} A - \frac{J_3}{J_4} B \right) \\
  \left( 1 - \frac{J_3^2}{J_2 J_4} \right) B &= -\frac{J_3}{2 J_2 J_4} + \left( \frac{I_2 J_3}{J_2 J_4} - \frac{I_1}{J_2} \right) A \\
  B &= \frac{\left( \frac{I_2 J_3}{J_2 J_4} - \frac{I_1}{J_2} \right) A - \frac{J_3}{2 J_2 J_4}}{\left( 1 - \frac{J_3^2}{J_2 J_4} \right)} \\
  &= \frac{\left( I_2 J_3 - I_1 J_4 \right) A - \frac{J_3}{2}}{J_2 J_4 - J_3^2} \\
  1 &= A I_0 + B J_1 + C J_2 \\
  &= A I_0 + B J_1 + \frac{J_2}{J_4}\left( \frac{1}{2} - A I_2 - B J_3 \right) \\
  1 - \frac{J_2}{2 J_4} &= \left( I_0 - \frac{I_2 J_2}{J_4} \right) A + \left( J_1 - \frac{J_2 J_3}{J_4} \right) B \\
  1 - \frac{J_2}{2 J_4} &= \left( I_0 - \frac{I_2 J_2}{J_4} \right) A - \frac{\left( J_1 - \frac{J_2 J_3}{J_4} \right) J_3}{2\left( J_2 J_4 - J_3^2 \right)} + \frac{\left( J_1 - \frac{J_2 J_3}{J_4} \right)\left( I_2 J_3 - I_1 J_4 \right)}{\left( J_2 J_4 - J_3^2 \right)} A \\
  \left( 1 - \frac{J_2}{2J_4} \right) \left( J_2 J_4 - J_3^2 \right) &= \left( J_2 J_4 - J_3^2 \right) \left( I_0 - \frac{I_2 J_2}{J_4} \right) A - \frac{\left( J_1 - \frac{J_2 J_3}{J_4} \right) J_3}{2} + \left( J_1 - \frac{J_2 J_3}{J_4} \right) \left( I_2 J_3 - I_1 J_4 \right) A \\
  \left( 1 - \frac{J_2}{2 J_4} \right)\left( J_2 J_4 - J_3^2 \right) + \frac{\left( J_1 - \frac{J_2 J_3}{J_4} \right) J_3}{2} &= \left[ \left( J_2 J_4 - J_3^2 \right)\left( I_0 - \frac{I_2 J_2}{J_4} \right) + \left( J_1 - \frac{J_2 J_3}{J_4} \right)\left( I_2 J_3 - I_1 J_4 \right) \right] A \\
  J_2 J_4 - \frac{J_2^2}{2} - J_3^2 + \cancel{\frac{J_2 J_3^2}{2 J_4}} + \frac{J_1 J_3}{2} - \cancel{\frac{J_2 J_3^2}{2 J_4}} &= \left[ I_0 J_2 J_4 - I_2 J_2^2 - I_0 J_3^2 + \cancel{\frac{I_2 J_2 J_3^2}{J_4}} + I_2 J_1 J_3 - I_1 J_1 J_4 - \cancel{\frac{I_2 J_2 J_3^2}{J_4}} + I_1 J_2 J_3 \right] A \\
  J_2 J_4 - \frac{J_2^2}{2} + J_3\left( \frac{J_1}{2} - J_3 \right) &= \left[ I_0\left( J_2 J_4 - J_3^2 \right) + I_1\left( J_2 J_3 - J_1 J_4 \right) + I_2\left( J_1 J_3 - J_2^2 \right) \right] A \\
  A &=\frac{J_2 J_4 - \frac{J_2^2}{2} + J_3\left( \frac{J_1}{2} - J_3 \right)}{I_0\left( J_2 J_4 - J_3^2 \right) + I_1 \left( J_2 J_3 - J_1 J_4 \right) + I_2\left( J_1 J_3 - J_2^2 \right)}
\end{align}
```

```@raw html
</details>
```

```math
\begin{align}
  C &= \frac{\frac{1}{2} - A I_2 - B J_3}{J_4}  \\
  B &= \frac{\frac{1}{2} J_3 + A (I_1 J_4 - I_2 J_3)}{J_3^2 - J_2 J_4} \\
  A &= \frac{J_3^2 - J_2 J_4 + \frac{1}{2} (J_2^2 - J_1 J_3)}{I_0 (J_3^2 - J_2 J_4) + I_1 (J_1 J_4 - J_2 J_3) + I_2 (J_2^2 - J_1 J_3)}
\end{align}
```

#### Evolving $u_\parallel$

When evolving only $u_\parallel$ and $n$ separately, we only need two
constraints. This corresponds to $C=0$ so that

```math
\begin{align}
  1 &= A I_0 + B J_1 \\
  0 &= A I_1 + B J_2 \\
  B &= -\frac{A I_1}{J_2} \\
  A I_0 &= 1 - B J_1 = 1 + \frac{A I_1 J_1}{J_2} \\
  A &= \frac{1}{I_0 - \frac{I_1 J_1}{J_2}}
\end{align}
```

#### Evolving $n$

When only evolving $n$ separately, the constraint is the same as in the bulk of
the domain

```math
\begin{align}
  1 &= AI_0 \\
  A &= \frac{1}{I_0}
\end{align}
```

Neutrals
--------

### Boundary conditions

Ions and neutrals that reach the wall are both recycled as neutrals. The
neutrals are emitted from the wall with a 'Knudsen cosine' distribution
characterised by a specified temperature $T_\mathrm{wall}$ (see Excalibur
report TN-05). The Knudsen distribution is given -- here assuming that the
magnetic field is perpendicular to the wall (so that $v_\parallel$ is the
velocity normal to the wall) -- by
```math
f_{Kw}(v_\zeta,v_r,v_z) = \frac{3}{4\pi} \left(\frac{m_i}{T_\mathrm{wall}}\right)^2 \frac{|v_z|}{\sqrt{v_\zeta^2 + v_r^2 + v_z^2}} \exp\left( -\frac{m_i(v_\zeta^2 + v_r^2 + v_z^2)}{2T_\mathrm{wall}} \right).
```
Note that $f_{Kw}$ is normalised so that it has unit flux
$\int d^3v\,|v_z| f_{Kw}(v_\zeta,v_r,v_z) = 1$.

The boundary condition for the neutrals at the lower target is then (for the
neutrals leaving whe wall)
```math
f_n(r,z=-\frac{L_z}{2},v_\zeta,v_r,v_z>0) = \Gamma_\mathrm{lower}(r) f_{Kw}(v_\zeta,v_r,|v_z|)
```
and at the upper target
```math
f_n(r,z=\frac{L_z}{2},v_\zeta,v_r,v_z<0) = \Gamma_\mathrm{upper}(r) f_{Kw}(v_\zeta,v_r,|v_z|).
```
A 'recycling fraction' is included, defined so that a fraction $0 \leq
R_\mathrm{recycle} \leq 1$ of the ions hitting the wall are recycled as
neutrals, while the whole flux of neutrals hitting the wall is always recycled.
(Recycling the 100% of the neutral flux means that the net flux of neutrals -
hitting the wall plus recycled - is $R_\mathrm{recycle}$ times the ion flux,
which makes applying boundary conditions in the moment-kinetic approach
simpler, see the next section.) This results in
```math
\begin{align}
  \Gamma_\mathrm{lower}(r) &= R_\mathrm{recycle} \frac{B_{z}}{B} 2\pi \int_{0}^{\infty} dv_{\perp} \int_{-\infty}^{0} dv_{\parallel}\, |v_{\parallel}| f_{i}(r,-L/2,v_{\perp},v_{\parallel}) \\
                           &\quad + \int dv_{\zeta}\,dv_{r} \int_{-\infty}^{0} dv_{z}\, |v_{z}| f_{n}(r,-L/2,v_{\zeta},v_{r},v_{z}) \\
  \Gamma_\mathrm{upper}(r) &= R_\mathrm{recycle} \frac{B_{z}}{B} 2\pi \int_{0}^{\infty} dv_{\perp} \int_{0}^{\infty} dv_{\parallel}\, |v_{\parallel}| f_{i}(r,L/2,v_{\perp},v_{\parallel}) \\
                           &\quad + \int dv_{\zeta}\,dv_{r} \int_{0}^{\infty} dv_{z}\, |v_{z}| f_{n}(r,L/2,v_{\zeta},v_{r},v_{z})
\end{align}
```

For 1D1V, we 'marginalise' -- i.e. integrate over $v_\perp$, assuming that
$v_\parallel=v_z$ (i.e. the magnetic field is perpendicular to the wall so
$B_{z}/B = 1$) -- (see Excalibur report TN-08) which gives
```math
\begin{align}
  f_{Kw,1V}(v_\parallel) &= \int dv_\zeta dv_r f_{Kw}(v_\zeta,v_r,v_\parallel) = 2\pi \int dv_\perp\,v_\perp f_{Kw}(v_\perp,v_\parallel) \\
                         &= 3\sqrt{\pi} \left(\frac{m_i}{2T_\mathrm{wall}}\right)^{3/2}|v_\parallel|\,\mathrm{erfc}\!\left(\sqrt{\frac{m_i}{2T_\mathrm{wall}}}|v_\parallel|\right)
\end{align}
```

### Moment constraints

When using the moment kinetic approach, we first need to apply a boundary
condition to the moments so that the net flux of neutrals leaving the wall
matches the recycling fraction $R_\mathrm{recycle}$ times the flux of ions reaching the wall
```math
\begin{align}
  u_{\parallel,n}(z=\pm L/2) = -R_\mathrm{recycle} \frac{n_{i}(z=\pm L/2) u_{\parallel,i}(z=\pm L/2)}{n_{n}(z=\pm L/2)}.
\end{align}
```
Having enforced the boundary condition on the flux, we need to impose that the
outgoing neutrals have the shape of a Knudsen cosine distribution, and ensure
that the constraints ([Constraints on normalized distribution function](@ref))
are satisfied. To impose three constraints we need three free parameters.
Taking as before the updated, incoming part of the neutral distribution
function before moment corrections to be
```math
\begin{align}
  \hat{g}_\mathrm{in}(w_\parallel) =
    \begin{cases}
      H(-w_{\parallel} v_{\mathrm{th},n} - u_{\parallel,n})\hat{g}(z,w_{\parallel}) & \text{at } z = -L/2 \\
      H(w_{\parallel} v_{\mathrm{th},n} + u_{\parallel,n})\hat{g}(z,w_{\parallel}) & \text{at } z = +L/2
    \end{cases}
\end{align}
```
and the shape for the Knudsen distribution to be
```math
\begin{align}
  \hat{g}_{Kw}(w_{\parallel}) =
    \begin{cases}
      H(w_{\parallel} v_{\mathrm{th},n} + u_{\parallel,n})f_{Kw,1V}(w_{\parallel} v_{\mathrm{th},n} + u_{\parallel,n}) & \text{at } z = -L/2 \\
      H(-w_{\parallel} v_{\mathrm{th},n} - u_{\parallel,n})f_{Kw,1V}(w_{\parallel} v_{\mathrm{th},n} + u_{\parallel,n}) & \text{at } z = +L/2
    \end{cases}
\end{align}
```
we define the final updated distribution function to be
```math
\begin{align}
  \tilde{g}_n(w_{\parallel}) = N_\mathrm{out} \hat{g}_{Kw} + N_\mathrm{in} \hat{g}_\mathrm{in} + C w_\parallel \hat{g}_\mathrm{in}
\end{align}
```
(note that if we chose to use $v_\parallel = w_\parallel v_{\mathrm{th},n} +
u_{\parallel,n}$ instead of $w_\parallel$ in the final term with the $C$
coefficient, this is just a shift by a constant and scale by another constant,
so would have the same form, just with different (but equivalent) values of the
$N_\mathrm{in}$ and $C$ coefficients).

Defining the integrals
```math
\begin{align}
  I_n = \int dw_{\parallel}\, w_{\parallel}^n \hat{g}_\mathrm{in}(w_{\parallel})
  K_n = \int dw_{\parallel}\, w_{\parallel}^n \hat{g}_{Kw}(w_{\parallel})
\end{align}
```
the constraints are
```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dw_{\|}\tilde{g}_{n}=1 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(N_\mathrm{out} \hat{g}_{Kw} + N_\mathrm{in} \hat{g}_\mathrm{in} + C w_\parallel \hat{g}_\mathrm{in}\right) \\
  &= N_\mathrm{out} K_{0} + N_\mathrm{in} I_{0} + C I_{1} \\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}\tilde{g}_{n}=0 & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(N_\mathrm{out} w_{\|} \hat{g}_{Kw} + N_\mathrm{in} w_{\|} \hat{g}_\mathrm{in} + C w_\parallel^2 \hat{g}_\mathrm{in}\right) \\
  &= N_\mathrm{out} K_{1} + N_\mathrm{in} I_{1} + C I_{2} \\
  \frac{1}{\sqrt{\pi}}\int dw_{\|}w_{\|}^{2}\tilde{g}_{n}=\frac{1}{2} & =\frac{1}{\sqrt{\pi}}\int dw_{\|}\left(N_\mathrm{out} w_{\|}^2 \hat{g}_{Kw} + N_\mathrm{in} w_{\|}^2 \hat{g}_\mathrm{in} + C w_\parallel^3 \hat{g}_\mathrm{in}\right) \\
  &= N_\mathrm{out} K_{2} + N_\mathrm{in} I_{2} + C I_{3}
\end{align}
```
which can be solved to find

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```

```math
\begin{align}
  C &= \frac{\left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right)}{I_{3}} \\
  N_\mathrm{out} &= \frac{\left(-N_\mathrm{in} I_{1} - C I_{2} \right)}{K_{1}} \\
                 &= -\frac{N_\mathrm{in} I_{1}}{K_{1}} - \frac{I_{2} \left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right)}{K_{1} I_{3}} \\
  N_\mathrm{out} K_{1} I_{3} &= -N_\mathrm{in} I_{1} I_{3} - I_{2} \left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right) \\
  N_\mathrm{out} &= -\frac{\left(N_\mathrm{in} \left(I_{1} I_{3} - I_{2}^2\right) + \frac{1}{2} I_{2}\right)}{\left(K_{1} I_{3} - K_{2} I_{2}\right)} \\
  N_\mathrm{in} &= \frac{\left(1 - N_\mathrm{out} K_{0} - C I_{1}\right)}{I_{0}} \\
                &= \frac{\left(1 - N_\mathrm{out} K_{0}\right)}{I_{0}} - \frac{C I_{1}}{I_{0}} \\
                &= \frac{\left(1 - N_\mathrm{out} K_{0}\right)}{I_{0}} - \frac{I_{1} \left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right)}{I_{0} I_{3}} \\
  N_\mathrm{in} I_{0} I_{3} &= \left(1 - N_\mathrm{out} K_{0}\right)I_{3} - I_{1} \left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right) \\
  N_\mathrm{in} \left(I_{0} I_{3} - I_{1} I_{2}\right) &= I_{3} - \frac{1}{2} I_{1} - N_\mathrm{out} \left(K_{0} I_{3} - I_{1} K_{2}\right) \\
  N_\mathrm{in} \left(I_{0} I_{3} - I_{1} I_{2}\right) &= I_{3} - \frac{1}{2} I_{1} + \left(K_{0} I_{3} - I_{1} K_{2}\right) \frac{\left(N_\mathrm{in} \left(I_{1} I_{3} - I_{2}^2\right) + \frac{1}{2} I_{2}\right)}{\left(K_{1} I_{3} - K_{2} I_{2}\right)} \\
  N_\mathrm{in} \left(I_{0} I_{3} - I_{1} I_{2}\right) \left(K_{1} I_{3} - K_{2} I_{2}\right) &= \left(I_{3} - \frac{1}{2} I_{1}\right) \left(K_{1} I_{3} - K_{2} I_{2}\right) + \left(K_{0} I_{3} - I_{1} K_{2}\right) \left(N_\mathrm{in} \left(I_{1} I_{3} - I_{2}^2\right) + \frac{1}{2} I_{2}\right) \\
  N_\mathrm{in} \left( \left(I_{0} I_{3} - I_{1} I_{2}\right) \left(K_{1} I_{3} - K_{2} I_{2}\right) - \left(K_{0} I_{3} - I_{1} K_{2}\right) \left(I_{1} I_{3} - I_{2}^2\right) \right) &= \left(I_{3} - \frac{1}{2} I_{1}\right) \left(K_{1} I_{3} - K_{2} I_{2}\right) + \left(K_{0} I_{3} - I_{1} K_{2}\right) \frac{1}{2} I_{2} \\
  N_\mathrm{in} \left( K_{0} I_{3} \left(I_{2}^2 - I_{1} I_{3}\right) + K_{1} I_{3} \left(I_{0} I_{3} - I_{1} I_{2}\right) + K_{2} \left(\cancel{I_{1} I_{2}^2} - I_{0} I_{2} I_{3} + I_{1}^2 I_{3} - \cancel{I_{1} I_{2}^2}\right)\right) &= \frac{1}{2} K_{0} I_{2} I_{3} + K_{1} I_{3} \left(I_3 - \frac{1}{2} I_{1}\right) + K_{2} \left(\cancel{\frac{1}{2} I_{1} I_{2}} - I_{2} I_{3} - \cancel{\frac{1}{2} I_{1} I_{2}} \right) \\
  N_\mathrm{in} \left( K_{0} \left(I_{2}^2 - I_{1} I_{3}\right) + K_{1} \left(I_{0} I_{3} - I_{1} I_{2}\right) + K_{2} \left(I_{1}^2 - I_{0} I_{2}\right)\right) &= \frac{1}{2} K_{0} I_{2} + K_{1} \left(I_3 - \frac{1}{2} I_{1}\right) - K_{2} I_{2} \\
  N_\mathrm{in} &= \frac{\left(\frac{1}{2} K_{0} I_{2} + K_{1} \left(I_3 - \frac{1}{2} I_{1}\right) - K_{2} I_{2}\right)}{\left( K_{0} \left(I_{2}^2 - I_{1} I_{3}\right) + K_{1} \left(I_{0} I_{3} - I_{1} I_{2}\right) + K_{2} \left(I_{1}^2 - I_{0} I_{2}\right)\right)} \\
\end{align}
```

```@raw html
</details>
```

```math
\begin{align}
  C &= \frac{\left(\frac{1}{2} - N_\mathrm{out} K_{2} - N_\mathrm{in} I_{2}\right)}{I_{3}} \\
  N_\mathrm{out} &= -\frac{\left(N_\mathrm{in} \left(I_{1} I_{3} - I_{2}^2\right) + \frac{1}{2} I_{2}\right)}{\left(K_{1} I_{3} - K_{2} I_{2}\right)} \\
  N_\mathrm{in} &= \frac{\left(\frac{1}{2} K_{0} I_{2} + K_{1} \left(I_3 - \frac{1}{2} I_{1}\right) - K_{2} I_{2}\right)}{\left( K_{0} \left(I_{2}^2 - I_{1} I_{3}\right) + K_{1} \left(I_{0} I_{3} - I_{1} I_{2}\right) + K_{2} \left(I_{1}^2 - I_{0} I_{2}\right)\right)}
\end{align}
```

#### Evolving $u_\parallel$

When evolving only $u_\parallel$ and $n$ separately, we only need two
constraints. This corresponds to $C=0$ so that
```math
\begin{align}
  N_\mathrm{out} = -\frac{I_{1}}{K_{1}} N_\mathrm{in} \\
  N_\mathrm{in} = \frac{1}{I_{0} - \frac{K_{0} I_{1}}{K_{1}}}
\end{align}
```

#### Evolving $n$

When only evolving $n$ separately, we still have $C=0$, but $N_\mathrm{in}$ and $N_\mathrm{out}$ must be adjusted to impose the density-moment constraint and the flux boundary condition.
```math
\begin{align}
  \frac{1}{\sqrt{\pi}}\int dv_{\|}\tilde{g}_{n}=1 &= \frac{1}{\sqrt{\pi}}\int dv_{\|}\left(N_\mathrm{out} \hat{g}_{Kw} + N_\mathrm{in} \hat{g}_\mathrm{in}\right) \\
  &= N_\mathrm{out} K_{0} + N_\mathrm{in} I_{0} \\
  \frac{1}{\sqrt{\pi}}\int dv_{\|}v_{\|}\tilde{g}_{n} = u_{n} &= \frac{1}{\sqrt{\pi}}\int dv_{\|}\left(N_\mathrm{out} v_{\|} \hat{g}_{Kw} + N_\mathrm{in} v_{\|} \hat{g}_\mathrm{in} \right) \\
  &= N_\mathrm{out} K_{1} + N_\mathrm{in} I_{1},
\end{align}
```
where $u_{n}$ is calculated from the ion flux as above, which can be solved to give
```math
\begin{align}
  N_\mathrm{out} &= \frac{\left(u_{n} - N_\mathrm{in} I_{1}\right)}{K_{1}} \\
  1 &= N_\mathrm{in} I_{0} + \frac{K_{0} \left(u_{n} - N_\mathrm{in} I_{1}\right)}{K_{1}} \\
  \Rightarrow N_\mathrm{in} &= \frac{\left(1 - \frac{K_{0} u_{n}}{K_{1}}\right)}{\left(I_{0} - \frac{K_{0} I_{1}}{K_{1}}\right)}
\end{align}
```
