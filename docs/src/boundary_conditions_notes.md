Wall boundary conditions with moment constraints
================================================

At the sheath entrance, the form of $f$ near $v_\|=0$ may be critical for
physical and/or numerical stability, e.g. how smoothly does it approach 0 at
$v_\|=0$? We therefore modify the algorithm  for applying the moment
constraints (described in [Constraints on normalized distribution
function](@ref))so that the corrections added drop smoothly to zero at
$v_\|=0$, hoping that this will lead to better behaviour. The corrections tend
to the same form as the one for the points not at the sheath boundary for
$v_\|>v_\mathrm{th}$. We use different algorithms for the ions and neutrals due
to their different sheath/wall boundary condiions.

Ion boundary condition and moment constraints
---------------------------------------------

The ion boundary condition at the sheath entrance is that the distribution
function vanishes for velocities that are outgoing from the wall
$f_i(z=z_\mathrm{wall}, v_\|\lessgtr 0)=0$.

To apply the moment constraints, given a normalized distribution function
produced by the time advance and after applying the boundary condition
$\hat{g}$, we define a new, corrected distribution function $\tilde{g}$ as

```math
\tilde{g} = A\hat{g}
            + B\frac{v_\|}{v_\mathrm{th}}\frac{|v_\| |/v_\mathrm{th}}{1+|v_\| |/v_\mathrm{th}}\hat{g}
            + C\left(\frac{v_\|}{v_\mathrm{th}}\right)^2\frac{|v_\| |/v_\mathrm{th}}{1+|v_\| |/v_\mathrm{th}}\hat{g}
```

!!! note

    JTO is not sure that this form for the moment corrections at the boundary
    is necessary, rather than using the simpler
    ```math
    \tilde{g} = A\hat{g} + Bw_\|\hat{g} + Cw_\|^2\hat{g}
    ```
    but the form described here was being used when the 1D1V
    `wall-bc_cheb_split3.toml` example finally ran, and is what is currently
    implemented.

Defining the integrals

```math
\begin{align}
I_n &= \int dw_\|w_\|^n\hat{g} \\
J_n &= \int dw_\|w_\|^n\frac{|v_\| |/v_\mathrm{th}}{1+|v_\| |/v_\mathrm{th}}\hat{g}
\end{align}
```

and noting $v_\|/v_\mathrm{th}=w_\|+u_\|/v_\mathrm{th}$ the constraints become

```math
\begin{align}
\int dw_\|\tilde{g} &= 1 =
    AI_0
    + B\left(J_1+\frac{u_\|}{v_\mathrm{th}}J_0\right)
    + C\left(J_2 + 2\frac{u_\|}{v_\mathrm{th}}J_1 + \frac{u_\|^2}{v_\mathrm{th}^2}J_0\right)\\
\int dw_\|w_\|\tilde{g} &= 0 =
    AI_1
    + B\left(J_2+\frac{u_\|}{v_\mathrm{th}}J_1\right)
    + C\left(J_3 + 2\frac{u_\|}{v_\mathrm{th}}J_2 + \frac{u_\|^2}{v_\mathrm{th}^2}J_1\right)\\
\int dw_\|w_\|^2\tilde{g} &= \frac{1}{2} =
    AI_2
    + B\left(J_3+\frac{u_\|}{v_\mathrm{th}}J_2\right)
    + C\left(J_4 + 2\frac{u_\|}{v_\mathrm{th}}J_3 + \frac{u_\|^2}{v_\mathrm{th}^2}J_2\right)\\
\end{align}
```
Defining $K_n=J_{n+1}+\frac{u_\|}{v_\mathrm{th}}J_n$,
$L_n=J_{n+2} + 2\frac{u_\|}{v_\mathrm{th}}J_{n+1} +
\frac{u_\|^2}{v_\mathrm{th}^2}J_n$ and solving the simultaneous equations,

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
AI_0 + BK_0 + CL_0 &= 1 \\
AI_1 + BK_1 + CL_1 &= 0 \\
AI_2 + BK_2 + CL_2 &= \frac{1}{2}
\end{align}
```
```math
\begin{align}
C &= \frac{\frac{1}{2} - AI_2 - BK_2}{L_2} \\
B &= -\frac{AI_1 + CL_1}{K_1} \\
  &= -\frac{I_1}{K_1}A - \frac{L_1}{K_1 L_2}\left( \frac{1}{2} - AI_2 - BK_2 \right) \\
\left( 1-\frac{K_2 L_1}{K_1 L_2} \right)B &= \frac{(I_2 L_1 - I_1 L_2)}{K_1 L_2}A - \frac{L_1}{2K_1 L_2} \\
(K_1 L_2 - K_2 L_1)B &= (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \\
AI_0 &= 1 - BK_0 - CL_0 \\
     &= 1
        - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} K_0 \\
     &\quad - \left(\frac{1}{2L_2} - \frac{I_2}{L_2}A - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)}\frac{K_2}{L_2} \right)L_0 \\
     &= 1
        - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} K_0 \\
     &\quad - \left(\frac{1}{2} - I_2A - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)}K_2 \right)\frac{L_0}{L_2} \\
     &= 1
        - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} K_0 \\
     &\quad - \frac{\left(\frac{1}{2}(K_1 L_2 - K_2 L_1) - I_2(K_1 L_2 - K_2 L_1)A - (I_2 L_1 - I_1 L_2)K_2A + \frac{L_1 K_2}{2} \right)}{(K_1 L_2 - K_2 L_1)} \frac{L_0}{L_2} \\
     &= 1
        - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} K_0 \\
     &\quad - \frac{\left(\frac{1}{2}K_1 L_2 - I_2 K_1 L_2 A + I_1 L_2 K_2 A \right)}{(K_1 L_2 - K_2 L_1)} \frac{L_0}{L_2} \\
     &= 1
        - \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} K_0
        - \frac{\left(\frac{1}{2}K_1 + (I_1 K_2 - I_2 K_1)A \right)}{(K_1 L_2 - K_2 L_1)} L_0 \\
\end{align}
```
```math
\begin{align}
& \left((K_1 L_2 - K_2 L_1)I_0 + (I_2 L_1 - I_1 L_2) K_0 + (I_1 K_2 - I_2 K_1) L_0 \right) A \\
& = (K_1 L_2 - K_2 L_1) + \frac{K_0 L_1}{2} - \frac{L_0 K_1}{2}
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
A &= \frac{K_1 L_2 - K_2 L_1 + \frac{1}{2}(K_0 L_1 - L_0 K_1)}{(K_1 L_2 - K_2 L_1) I_0 + (I_2 L_1 - I_1 L_2) K_0 + (I_1 K_2 - I_2 K_1) L_0} \\
%A &= \frac{\left( \left(L_2 - \frac{L_0}{2}\right)(K_1 L_2 - K_2 L_1) + (\frac{K_0 L_2}{2} - K_2 L_0) L_1 ) \right)}{\left( (L_2 - I_2)(K_1 L_2 - K_2 L_1) + (I_2 L_1 - I_1 L_2)(K_0 L_2 - K_2 L_0) \right)} \\
B &= \frac{\left( (I_2 L_1 - I_1 L_2)A - \frac{L_1}{2} \right)}{(K_1 L_2 - K_2 L_1)} \\
C &= \frac{\frac{1}{2} - AI_2 - BK_2}{L_2}
\end{align}
```

Neutral boundary condition and moment constraints
-------------------------------------------------

The neutral boundary condition is that all ions and neutrals incoming to the
wallare recycled as neutrals, and the neutrals are emitted from the wall with a
Knudsen cosine distribution (see eq. (7) of TN-08, "Numerical study of drift
kinetic model with wall BCs").
```math
F_\mathrm{Kw}(v_\|, v_\perp) = \frac{3}{\pi}\left( \frac{m_i}{2T_w} \right)^2
                               \frac{|v_\||}{\sqrt{v_\|^2+v_\perp^2}}
                               \exp\left( -\frac{m_i(v_\|^2+v_\perp^2)}{2T_w} \right)
```
For the 1D1V code where distribution functions are marginalised over
perpendicular velocities, this is (see eq. (12) of TN-08)
```math
f_\mathrm{Kw}(v_\|) \doteq 2\pi\int_0^\infty dv_\perp v_\perp F_\mathrm{Kw}(v_\|,v_\perp)
                    = 3\sqrt{\pi}\left( \frac{m_i}{2T_w} \right)^{3/2} |v_\||
                      \,\mathrm{erfc}\left( \sqrt{\frac{m_i}{2T_w}}|v_\|| \right)
```
For this boundary condition, there are two distinct (and non-zero) parts of the
distribution function, the incoming (to the wall) $f$ and the outgoing (from
the wall) Knudsen distribution.  We therefore adjust the overall amplitudes of
each component separately to give us two coefficients to apply constraints. For
the final constraint, we add one more coefficient that modifies only the
outgoing $f$ so that the incoming Knudsen distribution is not distorted.

Consider the upper boundary where the 'incoming' neutrals have $v_\|>0$ and the
'outgoing' Knudsen distribution is used for neutrals with $v_\|<0$ for
definiteness (for the other boundary just swap $v_\|<0 \leftrightarrow
v_\|>0$). Given the initial 'guess' $\hat{g}(v_\|>0)$ for the normalised
distribution function from the time solver, the corrected normalised
distribution function that obeys the constraints is
```math
\tilde{g}(w_\|(v_\|)) = \begin{cases}
    N_\mathrm{out} f_\mathrm{Kw}(v_\|)           & v_\| < 0 \\
    (N_\mathrm{in} + C w_\|)\hat{g}(w_\|(v_\|))  & v_\| > 0
\end{cases}
```

Defining the integrals
```math
\begin{align}
I_n &= \int_{-u_\|/v_\mathrm{th}}^\infty dw_\| w_\|^n \hat{g}(w_\|) \\
K_n &= \int_{-\infty}^{-u_\|/v_\mathrm{th}} dw_\| w_\|^n f_\mathrm{Kw}(v_\|(w_\|))
\end{align}
```
the constraints are
```math
\begin{align}
\int dw_\| \tilde{g} &= 1 = N_\mathrm{out} K_0 + N_\mathrm{in} I_0 + C I_1 \\
\int dw_\| w_\| \tilde{g} &= 0 = N_\mathrm{out} K_1 + N_\mathrm{in} I_1 + C I_2 \\
\int dw_\| w_\|^2 \tilde{g} &= \frac{1}{2} = N_\mathrm{out} K_2 + N_\mathrm{in} I_2 + C I_3 \\
\end{align}
```
Solving the simultaneous equations
```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
C &= \frac{\frac{1}{2} - N_\mathrm{out} K_2 - N_\mathrm{in} I_2}{I_3} \\
N_\mathrm{out} &= - \frac{N_\mathrm{in} I_1 + C I_2}{K_1} \\
               &= - \frac{N_\mathrm{in} I_1}{K_1}
                  - \frac{\frac{1}{2} - N_\mathrm{out} K_2 - N_\mathrm{in} I_2}{I_3} \frac{I_2}{K_1}\\
(I_3 K_1 - I_2 K_2) N_\mathrm{out} &= (I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2} \\
N_\mathrm{out} &= \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} \\
I_0 N_\mathrm{in} &= 1 - N_\mathrm{out} K_0 - C I_1 \\
                  &= 1
                     - \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} K_0 \\
              &\quad - \left(\frac{1}{2 I_3} - \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} \frac{K_2}{I_3} - N_\mathrm{in} \frac{I_2}{I_3} \right) I_1 \\
(I_0 I_3 - I_1 I_2) N_\mathrm{in} &= (I_3 - \frac{I_1}{2})
                                 - \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} I_3 K_0 \\
                          &\quad + \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} I_1 K_2 \\
(I_0 I_3 - I_1 I_2) N_\mathrm{in} &= (I_3 - \frac{I_1}{2})
                                 + \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} (I_1 K_2 - I_3 K_0) \\
\end{align}
```
```math
\begin{align}
& ((I_0 I_3 - I_1 I_2)(I_3 K_1 - I_2 K_2) - (I_2^2 - I_1 I_3)(I_1 K_2 - I_3 K_0)) N_\mathrm{in} \\
&= (I_3 - \frac{I_1}{2})(I_3 K_1 - I_2 K_2) - \frac{I_2}{2} (I_1 K_2 - I_3 K_0) \\

& (I_0 I_3^2 K_1 - I_0 I_2 I_3 K_2 - I_1 I_2 I_3 K_1 + I_2^2 I_3 K_0 + I_1^2 I_3 K_2 - I_1 I_3^2 K_0) N_\mathrm{in} \\
&= I_3^2 K_1 - I_2 I_3 K_2 - \frac{1}{2} I_1 I_3 K_1 + \frac{1}{2} I_2 I_3 K_0 \\
& (I_0 I_3 K_1 - I_0 I_2 K_2 - I_1 I_2 K_1 + I_2^2 K_0 + I_1^2 K_2 - I_1 I_3 K_0) N_\mathrm{in} \\
&= I_3 K_1 - I_2 K_2 - \frac{1}{2} I_1 K_1 + \frac{1}{2} I_2 K_0 \\
\end{align}
```
```@raw html
</details>
```

```math
\begin{align}
N_\mathrm{in} &= \frac{I_2 (\frac{1}{2} K_0 - K_2) + (I_3 - \frac{1}{2} I_1) K_1}{(I_2^2 - I_1 I_3) K_0 + (I_0 I_3 - I_1 I_2) K_1 + (I_1^2 - I_0 I_2) K_2} \\
N_\mathrm{out} &= \frac{(I_2^2 - I_1 I_3)N_\mathrm{in} - \frac{I_2}{2}}{I_3 K_1 - I_2 K_2} \\
C &= \frac{\frac{1}{2} - N_\mathrm{out} K_2 - N_\mathrm{in} I_2}{I_3} \\
\end{align}
```
