Wall boundary conditions with moment constraints
================================================

Boundary conditions
-------------------

The sheath-edge boundary conditions for the ions is that no ions leave from the sheath edge. So at the lower boundary $z=-L_z/2$

```math
\begin{align}
  f(z=-L/2,v_\parallel>0) = 0
\end{align}
```

and at the upper boundary $z=L_z/2

```math
\begin{align}
  f(z=L/2,v_\parallel<0) = 0
\end{align}
```

Moment constraints
------------------

At the sheath-entrance boundary, the constraints need to be enforced slightly differently to how they are done in the bulk of the domain (see [Constraints on normalized distribution function](@ref)). For compatibility with the boundary condition, the corrections which are added to impose the constraints should go to zero at $v_\parallel=0$. Note that the constraints are imposed after the boundary condition is applied by setting $f(v_\parallel>0)=0$ on the lower sheath boundary or $f(v_\parallel<0)=0$ on the upper sheath boundary.

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

### Evolving $u_\parallel$

When evolving only $u_\parallel$ and $n$ separately, we only need two constraints. This corresponds to $C=0$ so that

```math
\begin{align}
  1 &= A I_0 + B J_1 \\
  0 &= A I_1 + B J_2 \\
  B &= -\frac{A I_1}{J_2} \\
  A I_0 &= 1 - B J_1 = 1 + \frac{A I_1 J_1}{J_2} \\
  A &= \frac{1}{I_0 - \frac{I_1 J_1}{J_2}}
\end{align}
```

### Evolving $n$

When only evolving $n$ separately, the constraint is the same as in the bulk of the domain

```math
\begin{align}
  1 &= AI_0 \\
  A &= \frac{1}{I_0}
\end{align}
```
