Moment kinetic equations
========================

In 2D, assuming no variation in the $\zeta$ direction $\partial (\cdot) /
\partial \zeta = 0$, considering a tilted (or 'helical') magnetic field $\boldsymbol{B}
= B_z(r,z) \hat{\boldsymbol{z}} + B_\zeta(r,z) \hat{\boldsymbol{\zeta}}$, and including
'anomalous diffusion' in the radial direction, the ion kinetic equation for the
distribution function
$f_i(t, \boldsymbol{r}, \boldsymbol{v})=f_i(t, r, z, v_\parallel, v_\perp)$ is
```math
\begin{align}
\frac{\partial f_i}{\partial t}
    + v_E^r \frac{\partial f_i}{\partial r}
    + \left( v_E^z + b^z v_\parallel \right) \frac{\partial f_i}{\partial z}
    + \frac{E_y}{B} \frac{\partial f_i}{\partial r}
    - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial f_i}{\partial v_\parallel}
    = C_{ii}[f_i,f_i] + C_{in}[f_i,f_n] + S_i + D_r \frac{\partial^2 f_i}{\partial r^2}
\end{align}
```
where
```math
\begin{align}
b^z = \nabla z \cdot \boldsymbol{b} = (B_z/B)
\end{align}
```
is the $z$-projection of $\boldsymbol{b}$, and
```math
\begin{align}
v_E^r &= \nabla r \cdot \boldsymbol v_E
       = \nabla r \cdot \frac{\boldsymbol{b} \times \nabla \phi}{B}
       = - (\nabla r \times \nabla z) \cdot \boldsymbol{b} \frac{1}{B} \frac{\partial \phi}{\partial z}
       = - \hat{\boldsymbol{\zeta}} \cdot \boldsymbol{b} \frac{1}{B} \frac{\partial \phi}{\partial z}
       = - \frac{b_\zeta}{B} \frac{\partial \phi}{\partial z} \\

v_E^z &= \nabla z \cdot \boldsymbol v_E
       = \nabla z \cdot \frac{\boldsymbol{b} \times \nabla \phi}{B}
       = (\nabla r \times \nabla z) \cdot \boldsymbol{b} \frac{1}{B} \frac{\partial \phi}{\partial r}
       = \hat{\boldsymbol{\zeta}} \cdot \boldsymbol{b} \frac{1}{B} \frac{\partial \phi}{\partial r}
       = \frac{b_\zeta}{B} \frac{\partial \phi}{\partial r} \\
\end{align}
```
are the $r$- and $z$-components of the $E \times B$ drift.

The 1D version takes $\partial (\cdot) / \partial r = 0$, $B_\zeta = 0$,
$B_z = B$ so that $v_E^r = v_E^z = 0$, $b^z = 1$, $b_\zeta = 0$.

$\hat{\boldsymbol{z}} = \boldsymbol{b}$ is the parallel direction, $E_y$ is the
electric field in the binormal direction $\boldsymbol{\hat{y}} = \boldsymbol{b}
\times \hat{\boldsymbol{r}}$, $C_{ii}$ are ion-ion collisions, $C_{in}$
ion-neutral collisions/reactions, $S_i$ a source term, and $D_r$ is a constant
anomalous diffusion coefficient.

Simplified neutral interactions $C_{in}$ including charge exchange and
ionization with constant $R_\mathrm{CX}$ and $R_\mathrm{ioniz}$
```math
\begin{align}
C_{in} = -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n
\end{align}
```

Want to normalise the distribution for any species $s$ to extract low-order
moments.
```math
\begin{align}
F_s(t,z,w_\parallel,w_\perp) =
  \frac{v_{Ts}^3}{n_s} f_s(t, z, u_{s\parallel}(t,z) + v_{Ts}(t,z)w_\parallel, v_{Ts}(t,z)w_\perp)
\end{align}
```
with normalised velocities
```math
\begin{align}
w_\parallel(t,z,v_\parallel) &= \frac{v_\parallel - u_{s\parallel}(t,z)}{v_{Ts}(t,z)} \\
w_\perp(t,z,v_\perp) &= \frac{v_\perp}{v_{Ts}(t,z)}
\end{align}
```
the density
```math
\begin{align}
n_s(t,z) = 2\pi\int_{-\infty}^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp f_s(t,z,v_\parallel,v_\perp)
\end{align}
```
the average parallel velocity
```math
\begin{align}
u_{s\parallel}(t,z) = \frac{2\pi}{n_s}\int_{-\infty}^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp v_\parallel f_s(t,z,v_\parallel,v_\perp)
\end{align}
```
and the thermal speed
```math
\begin{align}
v_{Ts}^2(t,z) = \frac{4\pi}{3n_s}\int_{-\infty}^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp
    \left[ (v_\parallel - u_{s\parallel}(t,z))^2 + v_\perp^2 \right] f_s(t,z,v_\parallel,v_\perp)
\end{align}
```
For use later, we also give the definitions of the parallel and perpendicular pressure
```math
\begin{align}
p_{s\parallel}(t,z) &= 2\pi\int_{-\infty}^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp
    m_s \left( v_\parallel - u_{s\parallel}(t,z) \right)^2 f_s(t,z,v_\parallel,v_\perp) \nonumber \\
    &= m_s n_s v_{Ts}^2 2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp
        w_\parallel^2 F_s(t,z,w_\parallel,w_\perp) \\

p_{s\perp}(t,z) &= 2\pi\int_{-\infty}^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp
    m_s \frac{v_\perp^2}{2} f_s(t,z,v_\parallel,v_\perp) \\
    &= \frac{1}{2} m_s n_s v_{Ts}^2 2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp
        w_\perp^2 F_s(t,z,w_\parallel,w_\perp) \\
\end{align}
```
and $T_{s\parallel} = p_{s\parallel}/n_s$, $T_{s\perp} = p_{s\perp}/n_s$, which
we can note means that
```math
\begin{align}
p_s &= n_s T_s = \frac{m_s}{2} n_s v_{Ts}^2 = \frac{1}{3}(p_{s\parallel} + 2p_{s\perp}) \\
T_s &= \frac{1}{3}(T_{s\parallel} + 2T_{s\perp}) \\
v_{Ts} &= \sqrt{\frac{2T_s}{m_s}}= \sqrt{\frac{2(T_{s\parallel} + 2T_{s\perp})}{3m_s}} \\
\end{align}
```
The parallel heat flux is
```math
\begin{align}
q_{s\parallel} &= \int \frac{m_s}{2} \left( (v_\parallel - u_{s\parallel})^2 + v_\perp^2 \right) (v_\parallel - u_{s\parallel}) f_s d^3 v \\
    &= \frac{n_s}{v_{Ts}^3} \int \frac{m_s}{2} \left( (v_\parallel - u_{s\parallel})^2 + v_\perp^2 \right) (v_\parallel - u_{s\parallel}) F_s d^3 v \nonumber \\
    &= n_s \int \frac{m_s}{2} v_{Ts}^2 \left( w_\parallel^2 + w_\perp^2 \right) v_{Ts} w_\parallel F_s d^3 w \nonumber \\
    &= n_s v_{Ts}^3 \int \frac{m_s}{2} \left( w_\parallel^2 + w_\perp^2 \right) w_\parallel F_s d^3 w \\
\end{align}
```

 $F_s$ must therefore satisfy the conditions
```math
\begin{align}
2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp F_s(t,z,w_\parallel,w_\perp) &= 1 \\
2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp w_\parallel F_s(t,z,w_\parallel,w_\perp) &= 0 \\
2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp (w_\parallel^2 + w_\perp^2) F_s(t,z,w_\parallel,w_\perp) &= \frac{3}{2} \\
\end{align}
```

Ion moment equations
--------------------

Can integrate the drift kinetic equation to give the moment equations:
* continuity
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  ```math
  \begin{align}
  & \int \frac{\partial f_i}{\partial t} d^3 v
      + \underbrace{\int v_E^r \frac{\partial f_i}{\partial r} d^3 v}_\text{take prefactors and derivative out of velocity integral} \nonumber \\
  &   + \underbrace{\int v_E^z \frac{\partial f_i}{\partial z} d^3 v}_\text{take prefactors and derivative out of velocity integral}
  &   + \underbrace{\int v_\parallel b^z \frac{\partial f_i}{\partial z} d^3 v}_\text{take prefactor and derivative out of velocity integral} \nonumber \\
  &   - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \underbrace{\int \frac{\partial f_i}{\partial v_\parallel} d^3 v}_\text{total derivative integrates to 0} \nonumber \\
  &= \underbrace{\int C_{ii}[f_i,f_i] d^3 v}_{=0\text{, collisions conserve particles}} +
      \int \left[ \underbrace{-R_\mathrm{CX}(n_n f_i - n_i f_n)}_{=0\text{, no particle source from CX}} + R_\mathrm{ioniz} n_e f_n \right] d^3 v
      + \underbrace{\int S_i d^3 v}_{S_{i,n}} + D_r \frac{\partial^2}{\partial r^2} \int f_i d^3 v \\
  \end{align}
  ```
  ```@raw html
  </details>
  ```
  ```math
  \begin{align}
  \frac{\partial n_i}{\partial t}
      + v_E^r \frac{\partial n_i}{\partial r}
      + v_E^z \frac{\partial n_i}{\partial z}
      + b^z \frac{\partial}{\partial z}\left( n_i u_{i\parallel} \right)
      = R_\mathrm{ioniz} n_e n_n + S_{i,n} + D_r \frac{\partial^2 n_i}{\partial r^2} \\
  \end{align}
  ```
  where the density source is $S_{s,n} = \int S_s d^3 v$

* momentum
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  ```math
  \begin{align}
  & \int v_\parallel \frac{\partial f_i}{\partial t} d^3 v
      + \int v_\parallel v_E^r \frac{\partial f_i}{\partial r} d^3 v
      + \int v_\parallel v_E^z \frac{\partial f_i}{\partial z} d^3 v
      + \int v_\parallel^2 b^z \frac{\partial f_i}{\partial z} d^3 v
      - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int v_\parallel \frac{\partial f_i}{\partial v_\parallel} d^3 v \nonumber \\
  &= \underbrace{\int v_\parallel C_{ii}[f_i,f_i] d^3 v}_{=0\text{, collisions conserve momentum}} +
      \int v_\parallel \left[ -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n \right] d^3 v \nonumber \\
  &\quad+ \underbrace{\int v_\parallel S_i d^3 v}_{S_{i,\mathrm{mom}} / m_i}
      + D_r \frac{\partial^2}{\partial r^2} \int v_\parallel f_i d^3 v \\
  & \frac{\partial}{\partial t} \int v_\parallel f_i d^3 v
      + v_E^r \frac{\partial}{\partial r} \int v_\parallel f_i d^3 v
      + v_E^z \frac{\partial}{\partial z} \int v_\parallel f_i d^3 v
      + b^z \frac{\partial}{\partial z} \int v_\parallel^2 f_i d^3 v
      + b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int f_i d^3 v \nonumber \\
  &= \int v_\parallel \left[ -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n \right] d^3 v
      + \frac{1}{m_i} S_{i,\mathrm{mom}}
      + D_r \frac{\partial^2}{\partial r^2} \int v_\parallel f_i d^3 v \\
  & \frac{\partial}{\partial t} \int v_\parallel f_i d^3 v
      + v_E^r \frac{\partial}{\partial r} \int v_\parallel f_i d^3 v
      + v_E^z \frac{\partial}{\partial z} \int v_\parallel f_i d^3 v
      + b^z \frac{\partial}{\partial z}(u_{i\parallel}^2 n_i) + b^z \frac{\partial}{\partial z} \int v_{Ti}^2 \left( v_\parallel - u_{i\parallel} \right)^2 f_i d^3 v
      + b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int f_i d^3 v \nonumber \\
  &= \int v_\parallel \left[ -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n \right] d^3 v
      + \frac{1}{m_i} S_{i,\mathrm{mom}}
      + D_r \frac{\partial^2}{\partial r^2} \int v_\parallel f_i d^3 v \\
  \end{align}
  ```
  ```@raw html
  </details>
  ```
  ```math
  \begin{align}
  & m_i \frac{\partial}{\partial t}(n_i u_{i\parallel})
    + m_i v_E^r \frac{\partial}{\partial r} (n_i u_{i\parallel})
    + m_i v_E^z \frac{\partial}{\partial z} (n_i u_{i\parallel})
    + m_i b^z \frac{\partial}{\partial z}(n_i u_{i\parallel}^2)
    + b^z \frac{\partial p_{i\parallel}}{\partial z}
    + b^z e n_i \frac{\partial \phi}{\partial z} \nonumber \\
  &\quad= R_\mathrm{CX} m_i n_i n_n (u_{n\parallel} - u_{i\parallel})
          + R_\mathrm{ioniz} m_i n_e n_n u_{n\parallel}
          + S_{i,\mathrm{mom}}
          + m_i D_r \frac{\partial^2 (n_i u_{i\parallel})}{\partial r^2} \\
  \end{align}
  ```
  where the momentum source is
  $S_{s,\mathrm{mom}} = m_s \int v_\parallel S_s d^3 v$, which can also be
  manipulated into an equation for $\partial u_{i\parallel}/\partial t$ using
  the continuity equation
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  take $u_{i\parallel}\times$continuity $\rightarrow$
  ```math
  \begin{align}
  m_i u_{i\parallel} \frac{\partial n_i}{\partial t}
      + m_i u_{i\parallel} v_E^r \frac{\partial n_i}{\partial r}
      + m_i u_{i\parallel} v_E^z \frac{\partial n_i}{\partial z}
      + m_i u_{i\parallel} b^z \frac{\partial}{\partial z}(n_i u_{i\parallel})
  &= R_\mathrm{ioniz} m_i n_e n_n u_{i\parallel} + u_{i\parallel} S_{i,n}
     + m_i u_{i\parallel} D_r \frac{\partial^2 n_i}{\partial r^2} \\
  \end{align}
  ```
  ```math
  \begin{align}
  m_i u_{i\parallel} \frac{\partial n_i}{\partial t}
  &= - m_i u_{i\parallel} v_E^r \frac{\partial n_i}{\partial r}
     - m_i u_{i\parallel} v_E^z \frac{\partial n_i}{\partial z}
     - m_i u_{i\parallel} b^z \frac{\partial}{\partial z}(n_i u_{i\parallel})
     + R_\mathrm{ioniz} m_i n_e n_n u_{i\parallel} + u_{i\parallel} S_{i,n}
     + m_i u_{i\parallel} D_r \frac{\partial^2 n_i}{\partial r^2} \\
  \end{align}
  ```
  Momentum $\rightarrow$
  ```math
  \begin{align}
  &m_i u_{i\parallel} \frac{\partial n_i}{\partial t} + m_i n_i \frac{\partial u_{i\parallel}}{\partial t}
   + m_i n_i v_E^r \frac{\partial u_{i\parallel}}{\partial r}
   + m_i n_i v_E^r \frac{\partial n_i}{\partial r}
   + m_i n_i v_E^z \frac{\partial u_{i\parallel}}{\partial z}
   + m_i n_i v_E^z \frac{\partial n_i}{\partial z}
   + m_i b^z u_{i\parallel}^2 \frac{\partial n_i}{\partial z}
   + m_i b^z n_i 2u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
   + b^z \frac{\partial p_{i\parallel}}{\partial z}
   + b^z e n_i \frac{\partial\phi}{\partial z} \nonumber \\
  &= R_\mathrm{CX} m_i n_i n_n (u_{n\parallel} - u_{i\parallel})
     + R_\mathrm{ioniz} m_i n_e n_n u_{n\parallel}
     + S_{i,\mathrm{mom}}
     + m_i D_r \left( n_i \frac{\partial^2 u_{i\parallel}}{\partial r^2}
                      + 2 \frac{\partial n_i}{\partial r} \frac{\partial u_{i\parallel}}{\partial r}
                      + u_{i\parallel} \frac{\partial^2 n_i}{\partial r^2} \right) \\
  \end{align}
  ```
  Sub from continuity $\rightarrow$ cancellation
  ```@raw html
  </details>
  ```
  ```math
  \begin{align}
  &m_i n_i \frac{\partial u_{i\parallel}}{\partial t}
   + m_i n_i v_E^r \frac{\partial u_{i\parallel}}{\partial r}
   + m_i n_i v_E^z \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
   + m_i b^z n_i u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
  &= - b^z \frac{\partial p_{i\parallel}}{\partial z} - b^z e n_i \frac{\partial\phi}{\partial z}
     + R_\mathrm{CX} m_i n_i n_n (u_{n\parallel} - u_{i\parallel})
     + R_\mathrm{ioniz} m_i n_e n_n (u_{n\parallel} - u_{i\parallel}) \nonumber \\
  &\quad+ S_{i,\mathrm{mom}} - m_i u_{i\parallel} S_{i,n}
        + m_i D_r \left( n_i \frac{\partial^2 u_{i\parallel}}{\partial r^2}
                         + 2 \frac{\partial n_i}{\partial r} \frac{\partial u_{i\parallel}}{\partial r} \right) \\
  \end{align}
  ```
* Energy
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  ```math
  \begin{align}
  & \frac{1}{2} \int v^2 \frac{\partial f_i}{\partial t} d^3 v
      + \frac{1}{2} \int v^2 v_E^r f_i}{\partial r} d^3 v
      + \frac{1}{2} \int v^2 v_E^z f_i}{\partial z} d^3 v
      + \frac{1}{2} \int v^2 b^z v_\parallel \frac{\partial f_i}{\partial z} d^3 v
      - \frac{1}{2} b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int v^2 \frac{\partial f_i}{\partial v_\parallel} d^3 v \nonumber \\
  &\quad= \frac{1}{2} \underbrace{\int v^2 C_{ii}[f_i,f_i] d^3 v}_{=0\text{, collisions conserve energy}}
          + \frac{1}{2} \int v^2 \left[ -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n \right] d^3 v \nonumber \\
  &\qquad + \underbrace{\frac{1}{2} \int v^2 S_i d^3 v}_{S_{i,E} / m_i}
          + \frac{3}{2} D_r \frac{\partial^2}{\partial r^2} \int v^2 f_i d^3 v \\

  & \frac{1}{2} \frac{\partial}{\partial t} \int v^2 f_i d^3 v
      + \frac{1}{2} v_E^r r} \int v^2 f_i d^3 v
      + \frac{1}{2} v_E^z z} \int v^2 f_i d^3 v
      + \frac{1}{2} b^z \frac{\partial}{\partial z} \int v^2 v_\parallel f_i d^3 v
      + \frac{1}{2} b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int 2 v_\parallel f_i d^3 v \nonumber \\
  &\quad= \frac{1}{2} \int v^2 \left[ -R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n \right] d^3 v
          + \frac{1}{m_i} S_{i,E} \nonumber \\
  &\qquad + \frac{3}{2} D_r \frac{\partial^2}{\partial r^2} \int v^2 f_i d^3 v \\

  & \frac{1}{2} \frac{\partial}{\partial t} \int \left((v_\parallel - u_{i\parallel})^2 + \cancel{2(v_\parallel - u_{i\parallel}) u_{i\parallel}} + u_{i\parallel}^2 + v_\perp^2 \right) f_i d^3 v \nonumber \\
  &+ \frac{1}{2} v_E^r \frac{\partial}{\partial r} \int \left((v_\parallel - u_{i\parallel})^2 + \cancel{2(v_\parallel - u_{i\parallel}) u_{i\parallel}} + u_{i\parallel}^2 + v_\perp^2 \right) f_i d^3 v \nonumber \\
  &+ \frac{1}{2} v_E^z \frac{\partial}{\partial z} \int \left((v_\parallel - u_{i\parallel})^2 + \cancel{2(v_\parallel - u_{i\parallel}) u_{i\parallel}} + u_{i\parallel}^2 + v_\perp^2 \right) f_i d^3 v \nonumber \\
  &+ \frac{1}{2} b^z \frac{\partial}{\partial z} \int \underbrace{\left((v_\parallel - u_{i\parallel})^2 + 2(v_\parallel - u_{i\parallel}) u_{i\parallel} + u_{i\parallel}^2 + v_\perp^2 \right) \left( (v_\parallel - u_{i\parallel}) + u_{i\parallel} \right)}_{(v_\parallel - u_{i\parallel})^3 + 2(v_\parallel - u_{i\parallel})^2 u_{i\parallel} + \cancel{u_{i\parallel}^2(v_\parallel - u_{i\parallel})} + v_\perp^2(v_\parallel - u_{i\parallel}) + (v_\parallel - u_{i\parallel})^2 u_{i\parallel} + \cancel{2(v_\parallel - u_{i\parallel})u_{i\parallel}^2} + u_{i\parallel}^3 + v_\perp^2 u_{i\parallel}} f_i d^3 v \nonumber \\
  &+ \frac{1}{2} b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \int 2 \left( \cancel{(v_\parallel - u_{i\parallel})} + u_{i\parallel} \right) f_i d^3 v \nonumber \\
  &\quad= - \frac{1}{2} \int \left((v_\parallel - u_{i\parallel})^2 + \cancel{2(v_\parallel - u_{i\parallel}) u_{i\parallel}} + u_{i\parallel}^2 + v_\perp^2 \right) R_\mathrm{CX} n_n f_i d^3 v \nonumber \\
  &\qquad+ \frac{1}{2} \int \left((v_\parallel - u_{n\parallel})^2 + \cancel{2(v_\parallel - u_{n\parallel}) u_{n\parallel}} + u_{n\parallel}^2 + v_\perp^2 \right) R_\mathrm{CX} n_i f_n d^3 v \nonumber \\
  &\qquad+ \frac{1}{2} \int \left((v_\parallel - u_{n\parallel})^2 + \cancel{2(v_\parallel - u_{n\parallel}) u_{n\parallel}} + u_{n\parallel}^2 + v_\perp^2 \right) R_\mathrm{ioniz} n_e f_n d^3 v \nonumber \\
  &\qquad+ \frac{1}{m_i} S_{i,E} \nonumber \\
  &\qquad+ \frac{3}{2} D_r \frac{\partial^2}{\partial r^2} \int \left( (v_\parallel - u_{i\parallel})^2 + \cancel{2(v_\parallel - u_{i\parallel})} + u_{i\parallel}^2 + v_\perp^2 \right) f_i d^3 v \\

  & \frac{1}{2} \frac{\partial}{\partial t} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right) \nonumber \\
  &+ \frac{1}{2} v_E^r \frac{\partial}{\partial r} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right) \nonumber \\
  &+ \frac{1}{2} v_E^z \frac{\partial}{\partial z} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right) \nonumber \\
  &+ \frac{1}{2} b^z \frac{\partial}{\partial z} \left[ \int \left( (v_\parallel - u_{i\parallel})^3 + v_\perp^2(v_\parallel - u_{i\parallel}) \right) f_i d^3 v + 2\frac{p_{i\parallel}}{m_i} u_{i\parallel} + \frac{3 p_i}{m_i} u_{i\parallel} + n_i u_{i\parallel}^3 \right] \nonumber \\
  &+ b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} n_i u_{i\parallel} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(\frac{3 p_i}{m_i} n_n + n_i n_n u_{i\parallel}^2 - \frac{3 p_n}{m_i} n_i - n_i n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ \frac{1}{2} R_\mathrm{ioniz} n_e \left(\frac{3 p_n}{m_i} + n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ \frac{1}{m_i} S_{i,E} \nonumber \\
  &\qquad+ \frac{3}{2} D_r \frac{\partial^2}{\partial r^2} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right) \\

  & \frac{1}{2} \frac{\partial}{\partial t} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right)
    - \frac{1}{2} v_E^r \frac{\partial}{\partial r} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right)
    + \frac{1}{2} v_E^z \frac{\partial}{\partial z} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right)
    + \frac{1}{2} b^z \frac{\partial}{\partial z} \left[ \frac{2 q_{i\parallel}}{m_i} + 2\frac{p_{i\parallel}}{m_i} u_{i\parallel} + \frac{3 p_i}{m_i} u_{i\parallel} + n_i u_{i\parallel}^3 \right]
    + b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} n_i u_{i\parallel} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(\frac{3 p_i}{m_i} n_n + n_i n_n u_{i\parallel}^2 - \frac{3 p_n}{m_i} n_i - n_i n_n u_{n\parallel}^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e \left(\frac{3 p_n}{m_i} + n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ \frac{1}{m_i} S_{i,E}
        + \frac{3}{2} D_r \frac{\partial^2}{\partial r^2} \left( \frac{3 p_i}{m_i} + n_i u_{i\parallel}^2 \right) \\

  & \frac{3}{2} \frac{\partial p_i}{\partial t} + m_i u_{i\parallel} \frac{\partial(n_i u_{i\parallel})}{\partial t} - \frac{1}{2} m_i u_{i\parallel}^2 \frac{\partial n_i}{\partial t}
    + \frac{3}{2} v_E^r \frac{\partial p_i}{\partial r}
    + m_i n_i u_{i\parallel} v_E^r \frac{\partial u_{i\parallel}}{\partial r}
    + \frac{1}{2} m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r}
    + \frac{3}{2} v_E^z \frac{\partial p_i}{\partial z}
    + m_i n_i u_{i\parallel} v_E^z \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{1}{2} m_i u_{i\parallel}^2 v_E^z \frac{\partial n_i}{\partial z}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + u_{i\parallel} \frac{\partial p_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad+ \frac{3}{2} b^z u_{i\parallel} \frac{\partial p_i}{\partial z} + \frac{3}{2} b^z p_i \frac{\partial u_{i\parallel}}{\partial z} + \frac{3}{2} b^z m_i n_i u_{i\parallel}^2 \frac{\partial u_{i\parallel}}{\partial z} + \frac{1}{2} m_i u_{i\parallel}^3 \frac{\partial n_i}{\partial z}
    + b^z e \frac{\partial\phi}{\partial z} n_i u_{i\parallel} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(3 p_i n_n + m_i n_i n_n u_{i\parallel}^2 - 3 p_n n_i - m_i n_i n_n u_{n\parallel}^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e \left(3 p_n + m_i n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ \frac{1}{m_i} S_{i,E}
        + \frac{3}{2} D_r \left( 3 \frac{\partial^2 p_i}{\partial r^2} + m_i \frac{\partial^2 (n_i u_{i\parallel}^2)}{\partial r^2} \right) \\

  & \frac{3}{2} \frac{\partial p_i}{\partial t} \nonumber \\
  &\quad- m_i u_{i\parallel} v_E^r \frac{\partial}{\partial r} (n_i u_{i\parallel}) - b^z m_i u_{i\parallel} \frac{\partial}{\partial z}(n_i u_{i\parallel}^2) - m_i u_i{\parallel} v_E^z \frac{\partial}{\partial z} (n_i u_{i\parallel}) - b^z u_{i\parallel} \frac{\partial p_{i\parallel}}{\partial z} - b^z e n_i u_{i\parallel} \frac{\partial \phi}{\partial z} + u_{i\parallel} R_\mathrm{CX} m_i n_i n_n (u_{n\parallel} - u_{i\parallel}) + u_{i\parallel} R_\mathrm{ioniz} m_i n_e n_n u_{n\parallel} + u_{i\parallel} S_{i,\mathrm{mom}} + 3 m_i u_{i\parallel} D_r \frac{\partial^2 (n_i u_{i\parallel})}{\partial r^2} \nonumber \\
  &\quad+ \frac{1}{2} m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r} + \frac{1}{2} m_i u_{i\parallel}^2 v_E^z \frac{\partial n_i}{\partial z} + \frac{1}{2} b^z m_i u_{i\parallel}^2 \frac{\partial}{\partial z}(n_i u_{i\parallel}) - \frac{1}{2} m_i u_{i\parallel}^2 R_\mathrm{ioniz} n_e n_n - \frac{1}{2} m_i u_{i\parallel}^2 S_{i,n} - \frac{3}{2} m_i u_{i\parallel}^2 D_r \frac{\partial^2 n_i}{\partial r^2} \nonumber \\
  &+ \frac{3}{2} v_E^r \frac{\partial p_i}{\partial r}
    + m_i n_i u_{i\parallel} v_E^r \frac{\partial u_{i\parallel}}{\partial r}
    + \frac{1}{2} m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r} \nonumber \\
  &+ \frac{3}{2} v_E^z \frac{\partial p_i}{\partial z}
    + m_i n_i u_{i\parallel} v_E^z \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{1}{2} m_i u_{i\parallel}^2 v_E^z \frac{\partial n_i}{\partial z} \nonumber \\
  &\quad+ b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z u_{i\parallel} \frac{\partial p_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad+ b^z \frac{3}{2} u_{i\parallel} \frac{\partial p_i}{\partial z} + \frac{3}{2} b^z p_i \frac{\partial u_{i\parallel}}{\partial z} + \frac{3}{2} b^z m_i n_i u_{i\parallel}^2 \frac{\partial u_{i\parallel}}{\partial z} + \frac{1}{2} b^z m_i u_{i\parallel}^3 \frac{\partial n_i}{\partial z} \nonumber \\
  &+ b^z e \frac{\partial\phi}{\partial z} n_i u_{i\parallel} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(3 p_i n_n + m_i n_i n_n u_{i\parallel}^2 - 3 p_n n_i - m_i n_i n_n u_{n\parallel}^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e \left(3 p_n + m_i n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ S_{i,E}
        + \frac{3}{2} D_r \left( 3 \frac{\partial^2 p_i}{\partial r^2} + 2 m_i u_{i\parallel} \frac{\partial^2 (n_i u_{i\parallel})}{\partial r^2} + 2 m_i \frac{\partial u_{i\parallel}}{\partial r} \frac{\partial (n_i u_{i\parallel})}{\partial r} - m_i u_{i\parallel}^2 \frac{\partial^2 n_i}{\partial r^2} - 2 m_i u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial r} \frac{\partial n_i}{\partial r} \right) \\

  & \frac{3}{2} \frac{\partial p_i}{\partial t} \nonumber \\
  &\quad- \cancel{m_i n_i u_{i\parallel} v_E^r \frac{\partial u_{i\parallel}}{\partial r}} - \cancel{m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r}} - \cancel{m_i n_i u_i{\parallel} v_E^z \frac{\partial u_{i\parallel}}{\partial z}} - \cancel{m_i u_i{\parallel}^2 v_E^z \frac{\partial n_i}{\partial z}} - \cancel{2 b^z m_i n_i u_{i\parallel}^2 \frac{\partial u_{i\parallel}}{\partial z}} - \cancel{b^z m_i u_{i\parallel}^3 \frac{\partial n_i}{\partial z}} - \cancel{b^z u_{i\parallel} \frac{\partial p_{i\parallel}}{\partial z}} - \cancel{b^z e n_i u_{i\parallel} \frac{\partial \phi}{\partial z}} + u_{i\parallel} R_\mathrm{CX} m_i n_i n_n (u_{n\parallel} - u_{i\parallel}) + u_{i\parallel} R_\mathrm{ioniz} m_i n_e n_n u_{n\parallel} + u_{i\parallel} S_{i,\mathrm{mom}} + \cancel{3 m_i u_{i\parallel} D_r \frac{\partial^2 (n_i u_{i\parallel})}{\partial r^2}} \nonumber \\
  &\quad+ \cancel{\frac{1}{2} m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r}} + \cancel{\frac{1}{2} b^z m_i u_{i\parallel}^2 v_E^z \frac{\partial n_i}{\partial z}} + \cancel{\frac{1}{2} b^z m_i n_i u_{i\parallel}^2 \frac{\partial u_{i\parallel}}{\partial z}} + \cancel{\frac{1}{2} b^z m_i u_{i\parallel}^3 \frac{\partial n_i}{\partial z}} - \frac{1}{2} m_i u_{i\parallel}^2 R_\mathrm{ioniz} n_e n_n - \frac{1}{2} m_i u_{i\parallel}^2 S_{i,n} - \cancel{\frac{3}{2} m_i u_{i\parallel}^2 D_r \frac{\partial^2 n_i}{\partial r^2}} \nonumber \\
  &+ \frac{3}{2} v_E^r \frac{\partial p_i}{\partial r}
    + \cancel{m_i n_i u_{i\parallel} v_E^r \frac{\partial u_{i\parallel}}{\partial r}}
    + \cancel{\frac{1}{2} m_i u_{i\parallel}^2 v_E^r \frac{\partial n_i}{\partial r}} \nonumber \\
  &+ \frac{3}{2} v_E^z \frac{\partial p_i}{\partial z}
    + \cancel{m_i n_i u_{i\parallel} v_E^z \frac{\partial u_{i\parallel}}{\partial z}}
    + \cancel{\frac{1}{2} m_i u_{i\parallel}^2 v_E^z \frac{\partial n_i}{\partial z}} \nonumber \\
  &\quad+ b^z \frac{\partial q_{i\parallel}}{\partial z} + \cancel{b^z u_{i\parallel} \frac{\partial p_{i\parallel}}{\partial z}} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad+ \frac{3}{2} b^z u_{i\parallel} \frac{\partial p_i}{\partial z} + \frac{3}{2} b^z p_i \frac{\partial u_{i\parallel}}{\partial z} + \cancel{\frac{3}{2} b^z m_i n_i u_{i\parallel}^2 \frac{\partial u_{i\parallel}}{\partial z}} + \cancel{\frac{1}{2} b^z m_i u_{i\parallel}^3 \frac{\partial n_i}{\partial z}} \nonumber \\
  &+ \cancel{b^z e \frac{\partial\phi}{\partial z} n_i u_{i\parallel}} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(3 p_i n_n + m_i n_i n_n u_{i\parallel}^2 - 3 p_n n_i - m_i n_i n_n u_{n\parallel}^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e \left(3 p_n + m_i n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ S_{i,E}
        + \frac{3}{2} D_r \left( 3 \frac{\partial^2 p_i}{\partial r^2} + \cancel{2 m_i u_{i\parallel} \frac{\partial^2 (n_i u_{i\parallel})}{\partial r^2}} + 2 m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 + \cancel{2 m_i u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial r} \frac{\partial n_i}{\partial r}} - \cancel{m_i u_{i\parallel}^2 \frac{\partial^2 n_i}{\partial r^2}} - \cancel{2 m_i u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial r} \frac{\partial n_i}{\partial r}} \right) \\

  & \frac{3}{2} \frac{\partial p_i}{\partial t}
    + \frac{3}{2} v_E^r \frac{\partial p_i}{\partial r}
    + \frac{3}{2} v_E^z \frac{\partial p_i}{\partial z}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z u_{i\parallel} \frac{\partial p_i}{\partial z} + \frac{3}{2} b^z p_i \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} \left(3 p_i n_n - m_i n_i n_n u_{i\parallel}^2 + 2 m_i n_i n_n u_{i\parallel}u_{n\parallel} - 3 p_n n_i - m_i n_i n_n u_{n\parallel}^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e \left(3 p_n + m_i n_n u_{i\parallel}^2 - 2 m_i n_n u_{i\parallel} u_{n\parallel} + m_i n_n u_{n\parallel}^2 \right) \nonumber \\
  &\qquad+ S_{i,E}
         - u_{i\parallel} S_{i,\mathrm{mom}}
         + \frac{1}{2} m_i u_{i\parallel}^2 S_{i,n}
         + \frac{3}{2} D_r \left( 3 \frac{\partial^2 p_i}{\partial r^2} + 2 m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\
  \end{align}
  ```
  ```@raw html
  </details>
  ```

  ```math
  \begin{align}
  & \frac{3}{2} \frac{\partial p_i}{\partial t}
    + \frac{3}{2} v_E^r \frac{\partial p_i}{\partial r}
    + \frac{3}{2} v_E^z \frac{\partial p_i}{\partial z}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z u_{i\parallel} \frac{\partial p_i}{\partial z} + \frac{3}{2} b^z p_i \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p}
         + D_r \left( \frac{3}{2} \frac{\partial^2 p_i}{\partial r^2} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\
  \end{align}
  ```
  where the energy source is $S_{s,E} = \frac{1}{2} m_s \int v^2 S_s d^3 v$,
  and the pressure source is
  $S_{s,p} = \frac{1}{3} m_s \int |\boldsymbol{v} - u_{s\parallel}\hat{\boldsymbol{z}}|^2 S_s d^3 v = \frac{2}{3} S_{s,E} - \frac{2}{3} u_{s\parallel} S_{s,\mathrm{mom}} + \frac{1}{3} m_s u_{s\parallel}^2 S_{s,n}$.
  We use the pressure as an evolving variable in the code, so this is the
  energy equation used. It is also useful to subsitute in the continuity
  equation to convert this to a temperature equation and then a $v_{Ti}$
  equation, as the latter will be used to form the kinetic equation for $F_i$.
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  ```math
  \begin{align}
  & \frac{3}{2} \frac{\partial n_i T_i}{\partial t}
    + \frac{3}{2} v_E^r \frac{\partial n_i T_i}{\partial{r}}
    + \frac{3}{2} v_E^z \frac{\partial n_i T_i}{\partial{z}}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z u_{i\parallel} \frac{\partial n_i T_i}{\partial z} + \frac{3}{2} b^z n_i T_i \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p}
         + D_r \left( \frac{3}{2} \frac{\partial^2 (n_i T_i)}{\partial r^2} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\

  & \frac{3}{2} n_i \frac{\partial T_i}{\partial t} + \frac{3}{2} T_i \frac{\partial n_i}{\partial t}
    + \frac{3}{2} n_i v_E^r \frac{\partial T_i}{\partial{r}}
    + \frac{3}{2} T_i v_E^r \frac{\partial n_i}{\partial{r}}
    + \frac{3}{2} n_i v_E^z \frac{\partial T_i}{\partial{z}}
    + \frac{3}{2} T_i v_E^z \frac{\partial n_i}{\partial{z}}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z n_i u_{i\parallel} \frac{\partial T_i}{\partial z} + \frac{3}{2} b^z u_{i\parallel} T_i \frac{\partial n_i}{\partial z} + \frac{3}{2} b^z n_i T_i \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p}
         + D_r \left( \frac{3}{2} n_i \frac{\partial^2 T_i}{\partial r^2} + 3 \frac{\partial n_i}{\partial r} \frac{\partial T_i}{\partial r} + \frac{3}{2} T_i \frac{\partial^2 n_i}{\partial r^2} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\

  & \frac{3}{2} n_i \frac{\partial T_i}{\partial t}
    - \frac{3}{2} T_i v_E^r \frac{\partial n_i}{\partial r} - \frac{3}{2} T_i v_E^z \frac{\partial n_i}{\partial z} - \frac{3}{2} b^z n_i T_i \frac{\partial u_{i\parallel}}{\partial z} - \frac{3}{2} b^z u_{i\parallel} T_i \frac{\partial n_i}{\partial z} + \frac{3}{2} T_i R_\mathrm{ioniz} n_e n_n + \frac{3}{2} T_i S_{i,n} + \frac{3}{2} T_i D_r \frac{\partial^2 n_i}{\partial r^2}
    + \frac{3}{2} n_i v_E^r \frac{\partial T_i}{\partial{r}}
    + \frac{3}{2} T_i v_E^r \frac{\partial n_i}{\partial{r}}
    + \frac{3}{2} n_i v_E^z \frac{\partial T_i}{\partial{z}}
    + \frac{3}{2} T_i v_E^z \frac{\partial n_i}{\partial{z}}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z n_i u_{i\parallel} \frac{\partial T_i}{\partial z} + \frac{3}{2} b^z u_{i\parallel} T_i \frac{\partial n_i}{\partial z} + \frac{3}{2} b^z n_i T_i \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p}
         + D_r \left( \frac{3}{2} n_i \frac{\partial^2 T_i}{\partial r^2} + 3 \frac{\partial n_i}{\partial r} \frac{\partial T_i}{\partial r} + \frac{3}{2} T_i \frac{\partial^2 n_i}{\partial r^2} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\
  \end{align}
  ```
  ```@raw html
  </details>
  ```
  ```math
  \begin{align}
  & \frac{3}{2} n_i \frac{\partial T_i}{\partial t}
    + \frac{3}{2} n_i v_E^r \frac{\partial T_i}{\partial{r}}
    + \frac{3}{2} n_i v_E^z \frac{\partial T_i}{\partial{z}}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z n_i u_{i\parallel} \frac{\partial T_i}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p} - \frac{3}{2} T_i S_{i,n}
         + D_r \left( \frac{3}{2} n_i \frac{\partial^2 T_i}{\partial r^2} + 3 \frac{\partial n_i}{\partial r} \frac{\partial T_i}{\partial r} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\
  \end{align}
  ```
  ```@raw html
  <details>
  <summary style="text-align:center">[ intermediate steps ]</summary>
  ```
  ```math
  \begin{align}
  \frac{\partial T_i}{\partial t} &= \frac{1}{2} m_i \frac{\partial v_{Ti}^2}{\partial t} \nonumber \\
  &= m_i v_{Ti} \frac{\partial v_{Ti}}{\partial t} \\
  \frac{\partial T_i}{\partial z} &= \frac{1}{2} m_i \frac{\partial v_{Ti}^2}{\partial z} \nonumber \\
  &= m_i v_{Ti} \frac{\partial v_{Ti}}{\partial z} \\
  \frac{\partial T_i}{\partial r} &= \frac{1}{2} m_i \frac{\partial v_{Ti}^2}{\partial r} \nonumber \\
  &= m_i v_{Ti} \frac{\partial v_{Ti}}{\partial r} \\
  \end{align}
  ```
  ```math
  \begin{align}
  & \frac{3}{2} m_i n_i v_{Ti} \frac{\partial v_{Ti}}{\partial t}
    + \frac{3}{2} m_i n_i v_{Ti} v_E^r \frac{\partial v_{Ti}}{\partial{r}}
    + \frac{3}{2} m_i n_i v_{Ti} v_E^z \frac{\partial v_{Ti}}{\partial{z}}
    + b^z \frac{\partial q_{i\parallel}}{\partial z} + b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
    + \frac{3}{2} b^z m_i n_i u_{i\parallel} v_{Ti} \frac{\partial v_{Ti}}{\partial z} \nonumber \\
  &\quad= - \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p} - \frac{3}{2} T_i S_{i,n} \\
  \end{align}
  ```
  ```@raw html
  </details>
  ```
  ```math
  \begin{align}
  & \frac{3}{2} m_i n_i v_{Ti} \left( \frac{\partial v_{Ti}}{\partial t} + v_E^r \frac{\partial v_{Ti}}{\partial{r}} + v_E^z \frac{\partial v_{Ti}}{\partial{z}} + b^z u_{i\parallel} \frac{\partial v_{Ti}}{\partial z} \right) \nonumber \\
  &\quad= - b^z \frac{\partial q_{i\parallel}}{\partial z} - b^z p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}
       \nonumber \\
  &\qquad- \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
        + \frac{1}{2} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
  &\qquad+ \frac{3}{2} S_{i,p} - \frac{3}{2} T_i S_{i,n}
         + D_r \left( \frac{3}{2} n_i \frac{\partial^2 T_i}{\partial r^2} + 3 \frac{\partial n_i}{\partial r} \frac{\partial T_i}{\partial r} + m_i n_i \left( \frac{\partial u_{i\parallel}}{\partial r} \right)^2 \right) \\
  \end{align}
  ```

Ion kinetic equation
--------------------

Before giving the full 'moment kinetic' equation, we consider two reduced
versions which evolve separately only $n_i$, or only $n_i$ and $u_{i\parallel}$.

### Separate $n_i$

Normalise $n_i$ out of the distribution function. Velocity coordinates do not
need to be modified.
```math
\begin{align}
F_s(t,r,z,v_\parallel,v_\perp) = \frac{f_s(t,r,z,v_\parallel,v_\perp)}{n_s}
\end{align}
```
```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
& \frac{\partial n_i F_i}{\partial t} + v_E^r \frac{\partial n_i F_i}{\partial r} + (v_E^z + b^z v_\parallel) \frac{\partial n_i F_i}{\partial z}
    - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial n_i F_i}{\partial v_\parallel} \nonumber \\
&\quad= C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX}(n_n n_i F_i - n_i n_n F_n) + R_\mathrm{ioniz} n_e n_n F_n + S_i \\

& n_i \frac{\partial F_i}{\partial t} + F_i \frac{\partial n_i}{\partial t}
    + n_i v_E^r \frac{\partial F_i}{\partial r} + v_E^r F_i \frac{\partial n_i}{\partial r}
    + n_i (v_E^z + b^z v_\parallel) \frac{\partial F_i}{\partial z} + (v_E^z + b^z v_\parallel) F_i \frac{\partial n_i}{\partial z}
    - n_i b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial F_i}{\partial v_\parallel} \nonumber \\
&\quad= C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_i n_n (F_i - F_n) + R_\mathrm{ioniz} n_e n_n F_n + S_i \\

& n_i \frac{\partial F_i}{\partial t} - F_i v_E^r \frac{\partial n_i}{\partial r} - F_i v_E^z \frac{\partial n_i}{\partial z} - F_i b^z n_i \frac{\partial u_{i\parallel}}{\partial z} - F_i b^z u_{i\parallel} \frac{\partial n_i}{\partial z} + F_i R_\mathrm{ioniz} n_e n_n + F_i S_{i,n}
    + n_i v_E^r \frac{\partial F_i}{\partial r} + v_E^r F_i \frac{\partial n_i}{\partial r}
    + n_i (v_E^z + b^z v_\parallel) \frac{\partial F_i}{\partial z} + (v_E^z + b^z v_\parallel) F_i \frac{\partial n_i}{\partial z}
    - n_i b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial F_i}{\partial v_\parallel} \nonumber \\
&\quad= C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_i n_n (F_i - F_n) + R_\mathrm{ioniz} n_e n_n F_n + S_i \\
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
& \frac{\partial F_i}{\partial t}
  + v_E^r \frac{\partial F_i}{\partial r}
  + (v_E^z + b^z v_\parallel) \frac{\partial F_i}{\partial z}
  - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial F_i}{\partial v_\parallel}
  + \left( \frac{(b^z v_\parallel - b^z u_{i\parallel})}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\
\end{align}
```

### Separate $n_i$ and $u_{i\parallel}$

Normalise $n_i$ out of the distribution function. Shift to peculiar velocity.
```math
\begin{align}
F_s(t,r,z,\hat{w}_\parallel,v_\perp) &= \frac{f_s(t,r,z,\hat{w}_\parallel,v_\perp)}{n_s} \\
\hat{w}_\parallel &= v_\parallel - u_{s\parallel}
\end{align}
```
```math
\begin{align}
\left. \frac{\partial}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
  &= \left. \frac{\partial t}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial t} \right|_{r,z,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial r}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial z}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial \hat{w}_\parallel}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp}
    + \left. \frac{\partial v_\perp}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial v_\perp} \right|_{t,r,z,\hat{w}_\parallel} \\

  &= \left. \frac{\partial}{\partial t} \right|_{r,z,\hat{w}_\parallel,v_\perp}
    - \left. \frac{\partial u_{s\parallel}}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \\

  &\equiv \left. \frac{\partial}{\partial t} \right|_{r,z,\hat{w}_\parallel,v_\perp}
    - \frac{\partial u_{s\parallel}}{\partial t} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \\

\left. \frac{\partial}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
  &= \left. \frac{\partial t}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial r}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial z}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
    + \left. \frac{\partial \hat{w}_\parallel}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp}
    + \left. \frac{\partial v_\perp}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial v_\perp} \right|_{t,r,z,\hat{w}_\parallel} \\

  &= \left. \frac{\partial}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
    - \left. \frac{\partial u_{s\parallel}}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \\

  &\equiv \left. \frac{\partial}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
    - \frac{\partial u_{s\parallel}}{\partial z} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \\

\left. \frac{\partial}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
  &= \left. \frac{\partial}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
    - \frac{\partial u_{s\parallel}}{\partial r} \left. \frac{\partial}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \\
\end{align}
```
 $\partial / \partial v_\parallel |_{t,r,z,v_\perp} = \partial / \partial \hat{w}_\parallel |_{t,r,z,v_\perp}$
and
$\partial / \partial v_\perp |_{t,r,z,v_\parallel} = \partial / \partial v_\perp |_{t,r,z,\hat{w}_\parallel}$
as $u_{s\parallel}$ does not depend on $v_\parallel$ or $v_\perp$.

```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
The normalisation by $n_s$ to define $F_s$ is the same as for the 'separate
$n_i$' case, so we can start from the kinetic equation there and then transform
the coordinates to ${r,z,\hat{w}_\parallel,v_\perp,t}$
```math
\begin{align}
& \left. \frac{\partial F_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
  + v_E^r \left . \frac{\partial F_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
  + (v_E^z + b^z v_\parallel) \left . \frac{\partial F_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
  - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \left. \frac{\partial F_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
  + \left( \frac{(b^z v_\parallel - b^z u_{i\parallel})}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\

& \left. \frac{\partial F_i}{\partial t} \right|_{r,z,\hat{w}_\parallel,v_\perp}
  - \frac{\partial u_{i\parallel}}{\partial t} \left. \frac{\partial F_i}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \nonumber \\
& + v_E^r \left. \frac{\partial F_i}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
  - v_E^r \frac{\partial u_{i\parallel}}{\partial r} \left . \frac{\partial F_i}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \nonumber \\
& + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \left. \frac{\partial F_i}{\partial z} \right|_{t,r,\hat{w}_\parallel,v_\perp}
  - (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial u_{i\parallel}}{\partial z} \left . \frac{\partial F_i}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \nonumber \\
& - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \left. \frac{\partial F_i}{\partial \hat{w}_\parallel} \right|_{t,r,z,v_\perp} \nonumber \\
& + \left( \frac{b^z \hat{w}_\parallel}{n_i} \frac{\partial n_i}{\partial z} - \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\

& \frac{\partial F_i}{\partial t}
  + v_E^r \left. \frac{\partial F_i}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
  + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial F_i}{\partial z}
  - \left( \frac{\partial u_{i\parallel}}{\partial t}
           + v_E^r \frac{\partial u_{i\parallel}}{\partial r}
           + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial u_{i\parallel}}{\partial z} 
           + b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \right) \frac{\partial F_i}{\partial \hat{w}_\parallel} \nonumber \\
  &\quad+ \left( b^z \frac{\hat{w}_\parallel}{n_i} \frac{\partial n_i}{\partial z} - \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\
\end{align}
```
subsitute from the parallel flow equation
```math
\begin{align}
& \frac{\partial F_i}{\partial t}
  + v_E^r \left. \frac{\partial F_i}{\partial r} \right|_{t,z,\hat{w}_\parallel,v_\perp}
  + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial F_i}{\partial z}
  - \left( - \cancel{v_E^r \frac{\partial u_{i\parallel}}{\partial r}} - \cancel{v_E^z \frac{\partial u_{i\parallel}}{\partial z}} - \cancel{b^z u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}}
           - b^z \frac{1}{m_i n_i} \frac{\partial p_{i\parallel}}{\partial z}
           - \cancel{b^z \frac{e}{m_i} \frac{\partial \phi}{\partial z}}
           + R_\mathrm{CX} n_n (u_{n\parallel} - u_{i\parallel})
           + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} u_{n\parallel}
           + \frac{1}{m_i n_i} S_{i,\mathrm{mom}}
           - \frac{u_{i\parallel}}{n_i} S_{i,n}
           + \cancel{v_E^r \frac{\partial u_{i\parallel}}{\partial r}}
           + (\cancel{v_E^z} + b^z \hat{w}_\parallel + \cancel{b^z u_{i\parallel}}) \frac{\partial u_{i\parallel}}{\partial z}
           + \cancel{b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z}} \right) \frac{\partial F_i}{\partial \hat{w}_\parallel} \nonumber \\
  &\quad+ \left( b^z \frac{\hat{w}_\parallel}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
& \frac{\partial F_i}{\partial t}
  + v_E^r \frac{\partial F_i}{\partial r}
  + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial F_i}{\partial z} \nonumber \\
  &\quad- \left( b^z \hat{w}_\parallel \frac{\partial u_{i\parallel}}{\partial z}
                 - b^z \frac{1}{m_i n_i} \frac{\partial p_{i\parallel}}{\partial z}
                 + R_\mathrm{CX} n_n (u_{n\parallel} - u_{i\parallel})
                 + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} u_{n\parallel}
                 + \frac{1}{m_i n_i} S_{i,\mathrm{mom}}
                 - \frac{u_{i\parallel}}{n_i} S_{i,n}
               \right) \frac{\partial F_i}{\partial \hat{w}_\parallel} \nonumber \\
  &\quad+ \left( b^z \frac{\hat{w}_\parallel}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) F_i \nonumber \\
&\quad= \frac{1}{n_i} C_{ii}[n_i F_i, n_i F_i] - R_\mathrm{CX} n_n (F_i - F_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} F_n + \frac{1}{n_i} S_i \\
\end{align}
```

### [Full moment-kinetics (separate $n_i$, $u_{i\parallel}$ and $p_i$)](@id ion_full_moment_kinetic_equation)

Form evolution equation for $F_i(t,r,z,w_\parallel,w_\perp)$, starting from
kinetic equation for $f_i(t,r,z,v_\parallel,v_\perp)$
```math
\begin{align}
\left. \frac{\partial f_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
    + \dot{r}_i \left. \frac{\partial f_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
    + \dot{z}_i \left. \frac{\partial f_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
    + \dot{v}_{i\parallel} \left. \frac{\partial f_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
    + \dot{v}_{i\perp} \left. \frac{\partial f_i}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \nonumber \\
    \quad = C_{ii}[f_i, f_i] - R_\mathrm{CX}(n_n f_i - n_i f_n) + R_\mathrm{ioniz} n_e f_n + S_i
\end{align}
```
where
```math
\begin{align}
\dot{r}_i &= v_E^r \\
\dot{z}_i &= v_E^z + b^z v_\parallel \\
\dot{v}_{i\parallel} &= -b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \\
\dot{v}_{i\perp} &= 0 \\
\end{align}
```
and using the definitions of the normalised distribution function and
coordinates (repeated here for convenience)
```math
\begin{align}
F_s(t,r,z,w_\parallel,w_\perp) &=
  \frac{v_{Ts}^3}{n_s} f_s(t, r, z, u_{s\parallel}(t,z) + v_{Ts}(t,z)w_\parallel, v_{Ts}(t,z)w_\perp) \nonumber \\

w_\parallel(t,r,z,v_\parallel) &= \frac{v_\parallel - u_{s\parallel}(t,z)}{v_{Ts}(t,z)} \nonumber \\

w_\perp(t,r,z,v_\perp) &= \frac{v_\perp}{v_{Ts}(t,z)} \nonumber \\
\end{align}
```

Substituting the definition of $F_i$ gives
```math
\begin{align}
\left. \frac{\partial f_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
  &= \left. \frac{\partial}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left(\frac{n_i F_i}{v_{Ti}^3}\right) \nonumber \\
  &= \left. \frac{\partial n_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \frac{F_i}{v_{Ti}^3}
     - 3 \left. \frac{\partial v_{Ti}}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \frac{n_i F_i}{v_{Ti}^4}
     + \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \\

\dot{r}_i \left. \frac{\partial f_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
  &= \dot{r}_i \left. \frac{\partial}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left(\frac{n_i F_i}{v_{Ti}^3}\right) \nonumber \\
  &= \dot{r}_i \left. \frac{\partial n_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \frac{F_i}{v_{Ti}^3}
     - 3 \dot{r}_i \left. \frac{\partial v_{Ti}}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \frac{n_i F_i}{v_{Ti}^4}
     + \dot{r}_i \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \\

\dot{z}_i \left. \frac{\partial f_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
  &= \dot{z}_i \left. \frac{\partial}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left(\frac{n_i F_i}{v_{Ti}^3}\right) \nonumber \\
  &= \dot{z}_i \left. \frac{\partial n_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \frac{F_i}{v_{Ti}^3}
     - 3 \dot{z}_i \left. \frac{\partial v_{Ti}}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \frac{n_i F_i}{v_{Ti}^4}
     + \dot{z}_i \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \\

\dot{v}_{i\parallel} \left. \frac{\partial f_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
  &= \dot{v}_{i\parallel} \left. \frac{\partial}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left(\frac{n_i F_i}{v_{Ti}^3}\right) \nonumber \\
  &= \frac{n_i}{v_{Ti}^3} \dot{v}_{i\parallel} \left. \frac{\partial F_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \\

\dot{v}_{i\perp} \left. \frac{\partial f_i}{\partial v_\perp} \right|_{t,r,z,v_\parallel}
  &= \dot{v}_{i\perp} \left. \frac{\partial}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left(\frac{n_i F_i}{v_{Ti}^3}\right) \nonumber \\
  &= \frac{n_i}{v_{Ti}^3} \dot{v}_{i\perp} \left. \frac{\partial F_i}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \\
\end{align}
```
making the kinetic equation
```math
\begin{align}
&\frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
    + \dot{r}_i \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
    + \dot{z}_i \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
    + \dot{v}_{i\parallel} \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
    + \dot{v}_{i\perp} \frac{n_i}{v_{Ti}^3} \left. \frac{\partial F_i}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \nonumber \\
    &\quad + \left( \frac{1}{v_{Ti}^3} \frac{\partial n_i}{\partial t} - \frac{3 n_i}{v_{Ti}^4} \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \left( \frac{1}{v_{Ti}^3} \frac{\partial n_i}{\partial r} - \frac{3 n_i}{v_{Ti}^4} \frac{\partial v_{Ti}}{\partial r} \right) + \dot{z}_i \left( \frac{1}{v_{Ti}^3} \frac{\partial n_i}{\partial z} - \frac{3 n_i}{v_{Ti}^4} \frac{\partial v_{Ti}}{\partial z} \right) \right) F_i \nonumber \\
    &\quad= C_{ii}[\frac{n_i F_i}{v_{Ti}^3}, \frac{n_i F_i}{v_{Ti}^3}] - R_\mathrm{CX} \left( n_n \frac{n_i F_i}{v_{Ti}^3} - n_i \frac{n_n F_n}{v_{Tn}^3} \right) + R_\mathrm{ioniz} n_e \frac{n_n F_n}{v_{Tn}^3} + S_i \\

&\left. \frac{\partial F_i}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
    + \dot{r}_i \left. \frac{\partial F_i}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
    + \dot{z}_i \left. \frac{\partial F_i}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
    + \dot{v}_{i\parallel} \left. \frac{\partial F_i}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
    + \dot{v}_{i\perp} \left. \frac{\partial F_i}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \nonumber \\
    &\quad + \left( \frac{1}{n_i} \frac{\partial n_i}{\partial t} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \left( \frac{1}{n_i} \frac{\partial n_i}{\partial r} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \right) + \dot{z}_i \left( \frac{1}{n_i} \frac{\partial n_i}{\partial z} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \right) \right) F_i \nonumber \\
    &\quad= \frac{v_{Ti}^3}{n_i} C_{ii}[\frac{n_i F_i}{v_{Ti}^3}, \frac{n_i F_i}{v_{Ti}^3}] - R_\mathrm{CX} n_n \left( F_i - \frac{v_{Ti}^3}{v_{Tn}^3} F_n \right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \frac{v_{Ti}^3}{v_{Tn}^3} F_n + \frac{v_{Ti}^3}{n_i} S_i \\
\end{align}
```

The change of coordinates transforms the derivatives as
```math
\begin{align}
\left. \frac{\partial}{\partial t} \right|_{r,z,v_\parallel,v_\perp}
  &= \left. \frac{\partial t}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial t} \right|_{r,z,w_\parallel,w_\perp}
    + \left. \frac{\partial r}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,z,w_\parallel,w_\perp}
    + \left. \frac{\partial z}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial w_\parallel}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\parallel} \right|_{t,r,z,w_\perp}
    + \left. \frac{\partial w_\perp}{\partial t} \right|_{r,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\perp} \right|_{t,r,z,w_\parallel} \\
  &= \frac{\partial}{\partial t}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial t} - \frac{(v_\parallel - u_{i\parallel})}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial t} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{v_\perp}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial t} \frac{\partial}{\partial w_\perp} \\
  &= \frac{\partial}{\partial t}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial t} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} \frac{\partial}{\partial w_\perp} \\

\left. \frac{\partial}{\partial r} \right|_{t,z,v_\parallel,v_\perp}
  &= \left. \frac{\partial t}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial t} \right|_{r,z,w_\parallel,w_\perp}
    + \left. \frac{\partial r}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial z}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial w_\parallel}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\parallel} \right|_{t,r,z,w_\perp}
    + \left. \frac{\partial w_\perp}{\partial r} \right|_{t,z,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\perp} \right|_{t,r,z,w_\parallel} \\
  &= \frac{\partial}{\partial r}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial r} - \frac{(v_\parallel - u_{i\parallel})}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial r} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{v_\perp}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial r} \frac{\partial}{\partial w_\perp} \\
  &= \frac{\partial}{\partial r}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial r} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \frac{\partial}{\partial w_\perp} \\

\left. \frac{\partial}{\partial z} \right|_{t,r,v_\parallel,v_\perp}
  &= \left. \frac{\partial t}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial t} \right|_{r,z,w_\parallel,w_\perp}
    + \left. \frac{\partial r}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,z,w_\parallel,w_\perp}
    + \left. \frac{\partial z}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial w_\parallel}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\parallel} \right|_{t,r,z,w_\perp}
    + \left. \frac{\partial w_\perp}{\partial z} \right|_{t,r,v_\parallel,v_\perp} \left. \frac{\partial}{\partial w_\perp} \right|_{t,r,z,w_\parallel} \\
  &= \frac{\partial}{\partial z}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial z} - \frac{(v_\parallel - u_{i\parallel})}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial z} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{v_\perp}{v_{Ti}^2} \frac{\partial v_{Ti}}{\partial z} \frac{\partial}{\partial w_\perp} \\
  &= \frac{\partial}{\partial z}
    + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial z} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \right) \frac{\partial}{\partial w_\parallel}
    - \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \frac{\partial}{\partial w_\perp} \\

\left. \frac{\partial}{\partial v_\parallel} \right|_{t,r,z,v_\perp}
  &= \left. \frac{\partial t}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left. \frac{\partial}{\partial t} \right|_{r,z,w_\parallel,w_\perp}
    + \left. \frac{\partial r}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left. \frac{\partial}{\partial r} \right|_{t,z,w_\parallel,w_\perp}
    + \left. \frac{\partial z}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left. \frac{\partial}{\partial z} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial w_\parallel}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left. \frac{\partial}{\partial w_\parallel} \right|_{t,r,z,w_\perp}
    + \left. \frac{\partial w_\perp}{\partial v_\parallel} \right|_{t,r,z,v_\perp} \left. \frac{\partial}{\partial w_\perp} \right|_{t,r,z,w_\parallel} \\
  &= \frac{1}{v_{Ti}} \frac{\partial}{\partial w_\parallel} \\

\left. \frac{\partial}{\partial v_\perp} \right|_{t,r,z,v_\parallel}
  &= \left. \frac{\partial t}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left. \frac{\partial}{\partial t} \right|_{r,z,w_\parallel,w_\perp}
    + \left. \frac{\partial r}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left. \frac{\partial}{\partial r} \right|_{t,z,w_\parallel,w_\perp}
    + \left. \frac{\partial z}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left. \frac{\partial}{\partial z} \right|_{t,r,w_\parallel,w_\perp}
    + \left. \frac{\partial w_\parallel}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left. \frac{\partial}{\partial w_\parallel} \right|_{t,r,z,w_\perp}
    + \left. \frac{\partial w_\perp}{\partial v_\perp} \right|_{t,r,z,v_\parallel} \left. \frac{\partial}{\partial w_\perp} \right|_{t,r,z,w_\parallel} \\
  &= \frac{1}{v_{Ti}} \frac{\partial}{\partial w_\perp} \\
\end{align}
```
and so the kinetic equation becomes
```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
&\frac{\partial F_i}{\partial t} 
           + \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial t} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} \right) \frac{\partial F_i}{\partial w_\parallel}
           - \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} \frac{\partial F_i}{\partial w_\perp} \nonumber \\
    &\quad + \dot{r}_i \frac{\partial F_i}{\partial r}
           + \dot{r}_i \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial r} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \right) \frac{\partial F_i}{\partial w_\parallel}
           - \dot{r}_i \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \frac{\partial F_i}{\partial w_\perp} \nonumber \\
    &\quad + \dot{z}_i \frac{\partial F_i}{\partial z}
           + \dot{z}_i \left( -\frac{1}{v_{Ti}} \frac{\partial u_{i\parallel}}{\partial z} - \frac{w_\parallel}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \right) \frac{\partial F_i}{\partial w_\parallel}
           - \dot{z}_i \frac{w_\perp}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \frac{\partial F_i}{\partial w_\perp} \nonumber \\
    &\quad + \frac{\dot{v}_{i\parallel}}{v_{Ti}} \frac{\partial F_i}{\partial w_\parallel} \nonumber \\
    &\quad + \frac{\dot{v}_{i\perp}}{v_{Ti} \frac{\partial F_i}{\partial w_\perp} \nonumber \\
    &\quad + \left( \frac{1}{n_i} \frac{\partial n_i}{\partial t} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \left( \frac{1}{n_i} \frac{\partial n_i}{\partial r} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial r} \right) + \dot{z}_i \left( \frac{1}{n_i} \frac{\partial n_i}{\partial z} - \frac{3}{v_{Ti}} \frac{\partial v_{Ti}}{\partial z} \right) \right) F_i \nonumber \\
    &\quad= \frac{v_{Ti}^3}{n_i} C_{ii}[\frac{n_i F_i}{v_{Ti}^3}, \frac{n_i F_i}{v_{Ti}^3}] - R_\mathrm{CX} n_n \left( F_i - \frac{v_{Ti}^3}{v_{Tn}^3} F_n \right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \frac{v_{Ti}^3}{v_{Tn}^3} F_n + \frac{v_{Ti}^3}{n_i} S_i \\
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
&\frac{\partial F_i}{\partial t} + \dot{r}_i \frac{\partial F_i}{\partial r} + \dot{z}_i \frac{\partial F_i}{\partial z} \nonumber \\
    &\quad + \left( \frac{\dot{v}_{i\parallel}}{v_{Ti}} - \frac{1}{v_{Ti}} \left( \frac{\partial u_{i\parallel}}{\partial t} + \dot{r}_i \frac{\partial u_{i\parallel}}{\partial r} + \dot{z}_i \frac{\partial u_{i\parallel}}{\partial z} \right) \right. \nonumber \\
    &\qquad\quad  \left. - \frac{w_\parallel}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \frac{\partial v_{Ti}}{\partial r} + \dot{z}_i \frac{\partial v_{Ti}}{\partial z} \right) \right) \frac{\partial F_i}{\partial w_\parallel} \nonumber \\
    &\quad + \left( \frac{\dot{v}_{i\perp}}{v_{Ti}} - \frac{w_\perp}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \frac{\partial v_{Ti}}{\partial r} + \dot{z}_i \frac{\partial v_{Ti}}{\partial z} \right) \right) \frac{\partial F_i}{\partial w_\perp} \nonumber \\
    &\quad + \left( \frac{1}{n_i} \left( \frac{\partial n_i}{\partial t} + \dot{r}_i \frac{\partial n_i}{\partial r} + \dot{z}_i \frac{\partial n_i}{\partial z} \right) - \frac{3}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r}_i \frac{\partial v_{Ti}}{\partial r} + \dot{z}_i \frac{\partial v_{Ti}}{\partial z} \right) \right) F_i \nonumber \\
    &\quad= \frac{v_{Ti}^3}{n_i} C_{ii}[\frac{n_i F_i}{v_{Ti}^3}, \frac{n_i F_i}{v_{Ti}^3}] - R_\mathrm{CX} n_n \left( F_i - \frac{v_{Ti}^3}{v_{Tn}^3} F_n \right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \frac{v_{Ti}^3}{v_{Tn}^3} F_n + \frac{v_{Ti}^3}{n_i} S_i \\

&\frac{\partial F_i}{\partial t} + \dot{r} \frac{\partial F_i}{\partial r} + \dot{z} \frac{\partial F_i}{\partial z} + \dot{w}_\parallel \frac{\partial F_i}{\partial w_\parallel} + \dot{w}_\perp \frac{\partial F_i}{\partial w_\perp}
    = \dot{F}_i + \mathcal{C}_i + \frac{v_{Ti}^3}{n_i} S_i \\
\end{align}
```
where
```math
\begin{align}
\dot{r} &= v_E^r \\

\dot{z} &= v_E^z + b^z v_\parallel = v_E^z + b^z v_{Ti} w_\parallel + b^z u_{i\parallel} \\

\dot{w}_\parallel &= \frac{\dot{v}_{i\parallel}}{v_{Ti}} - \frac{1}{v_{Ti}} \left( \frac{\partial u_{i\parallel}}{\partial t} + \dot{r} \frac{\partial u_{i\parallel}}{\partial r} + \dot{z} \frac{\partial u_{i\parallel}}{\partial z} \right)
    - \frac{w_\parallel}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right) \\

\dot{w}_\perp &= \frac{\dot{v}_{i\perp}}{v_{Ti}} - \frac{w_\perp}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right) \\

\frac{\dot{F}_i}{F_i} &= \frac{3}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right) - \frac{1}{n_i} \left( \frac{\partial n_i}{\partial t} + \dot{r} \frac{\partial n_i}{\partial r} + \dot{z} \frac{\partial n_i}{\partial z} \right) \\

\mathcal{C}_i &= \frac{v_{Ti}^3}{n_i} C_{ii}[\frac{n_i F_i}{v_{Ti}^3}, \frac{n_i F_i}{v_{Ti}^3}] - R_\mathrm{CX} n_n \left( F_i - \frac{v_{Ti}^3}{v_{Tn}^3} F_n \right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \frac{v_{Ti}^3}{v_{Tn}^3} F_n \\
\end{align}
```
We could substitute in the moment equations to eliminate the
time-derivative-of-moment terms
```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
&\frac{\dot{v}_{i\parallel}}{v_{Ti}} - \frac{1}{v_{Ti}} \left( \frac{\partial u_{i\parallel}}{\partial t} + \dot{r}_i \frac{\partial u_{i\parallel}}{\partial r} + \dot{z} \frac{\partial u_{i\parallel}}{\partial z} \right) \nonumber \\
&\quad= - \cancel{b^z \frac{e}{m_i v_{Ti}} \frac{\partial \phi}{\partial z}} \nonumber \\
&\qquad - \frac{1}{v_{Ti}} \left( - \cancel{v_E^r \frac{\partial u_{i\parallel}}{\partial r}} - \cancel{v_E^z \frac{\partial u_{i\parallel}}{\partial z}} - \cancel{b^z u_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z}} - b^z \frac{1}{m_i n_i} \frac{\partial p_{i\parallel}}{\partial z} - \cancel{b^z \frac{e}{m_i} \frac{\partial \phi}{\partial z}} + R_\mathrm{CX} n_n \left(u_{n\parallel} - u_{i\parallel}\right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \left(u_{n\parallel} - u_{i\parallel}\right) + \frac{S_{i,\mathrm{mom}}}{m_i n_i} - \frac{u_{i\parallel} S_{i,n}}{n_i} \right) \nonumber \\
&\qquad - \cancel{\frac{1}{v_{Ti}} v_E^r \frac{\partial u_{i\parallel}}{\partial r}} \nonumber \\
&\qquad - \frac{1}{v_{Ti}} \left( \cancel{v_E^z} + b^z v_{Ti} w_\parallel + \cancel{b^z u_{i\parallel}} \right) \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
&\quad= - b^z w_\parallel \frac{\partial u_{i\parallel}}{\partial z} + b^z \frac{1}{v_{Ti}} \frac{\partial p_{i\parallel}}{\partial z} - \frac{1}{v_{Ti}} R_\mathrm{CX} n_n \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{1}{v_{Ti}} R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{S_{i,\mathrm{mom}}}{m_i n_iv_{Ti}} + \frac{u_{i\parallel} S_{i,n}}{n_i v_{Ti}} \\

&\frac{1}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r}\frac{\partial v_{Ti}}{\partial r} + \dot{z}\frac{\partial v_{Ti}}{\partial z} \right) \nonumber \\
&\quad= \frac{1}{v_{Ti}} \left( - \cancel{v_E^r \frac{\partial v_{Ti}}{\partial r}} - \cancel{v_E^z \frac{\partial v_{Ti}}{\partial z}} - \cancel{b^z u_{i\parallel} \frac{\partial v_{Ti}}{\partial z}} - \frac{2 b^z}{3 m_i n_i v_{Ti}} \frac{\partial q_{i\parallel}}{\partial z} - \frac{2 b^z}{3 m_i n_i v_{Ti}} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} - \frac{1}{3 m_i v_{Ti}} R_\mathrm{CX} n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) + \frac{1}{3 m_i n_i v_{Ti}} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) + \frac{S_{i,p}}{m_i n_i v_{Ti}} - \frac{T_i S_{i,n}}{m_i n_i v_{Ti}} \right) \nonumber \\
&\qquad + \cancel{\frac{1}{v_{Ti}} v_E^r \frac{\partial v_{Ti}}{\partial r}} \nonumber \\
&\qquad + \frac{1}{v_{Ti}} \left( \cancel{v_E^z} + b^z v_{Ti} w_\parallel + \cancel{b^z u_{i\parallel}} \right) \frac{\partial v_{Ti}}{\partial z} \nonumber \\
&\quad= b^z w_\parallel \frac{\partial v_{Ti}}{\partial z} - \frac{b^z}{3 p_i} \frac{\partial q_{i\parallel}}{\partial z} - \frac{b^z}{3 p_i} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} - \frac{1}{6 T_i} R_\mathrm{CX} n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) + \frac{1}{6 p_i} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) + \frac{S_{i,p}}{2 p_i} - \frac{S_{i,n}}{2 n_i} \nonumber \\

&\frac{1}{n_i} \left( \frac{\partial n_i}{\partial t} + \dot{r}\frac{\partial n_i}{\partial r} + \dot{z} \frac{\partial n_i}{\partial z} \right) \nonumber \\
&\quad= \frac{1}{n_i} \left( - \cancel{v_E^r \frac{\partial n_i}{\partial r}} - \cancel{v_E^z \frac{\partial n_i}{\partial z}} - b^z n_i \frac{\partial u_{i\parallel}}{\partial z} - \cancel{b^z u_{i\parallel} \frac{\partial n_i}{\partial z}} + R_\mathrm{ioniz} n_e n_n + S_{i,n}
                               + \cancel{v_E^r \frac{\partial n_i}{\partial r}}
                               + \left( \cancel{v_E^z} + b^z v_{Ti} w_\parallel + \cancel{b^z u_{i\parallel}} \right) \frac{\partial n_i}{\partial z}
                        \right) \nonumber \\
&\quad= b^z \frac{v_{Ti} w_\parallel}{n_i} \frac{\partial n}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{S_{i,n}}{n_i} \\

\dot{w}_\parallel &= - b^z w_\parallel \frac{\partial u_{i\parallel}}{\partial z} + b^z \frac{1}{v_{Ti}} \frac{\partial p_{i\parallel}}{\partial z} - \frac{1}{v_{Ti}} R_\mathrm{CX} n_n \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{1}{v_{Ti}} R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{S_{i,\mathrm{mom}}}{m_i n_iv_{Ti}} + \frac{u_{i\parallel} S_{i,n}}{n_i v_{Ti}} \nonumber \\
&\quad - b^z w_\parallel^2 \frac{\partial v_{Ti}}{\partial z} + \frac{b^z w_\parallel}{3 p_i} \frac{\partial q_{i\parallel}}{\partial z} + \frac{b^z w_\parallel}{3 p_i} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} + \frac{w_\parallel}{6 T_i} R_\mathrm{CX} n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) - \frac{w_\parallel}{6 p_i} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) - \frac{w_\parallel S_{i,p}}{2 p_i} + \frac{w_\parallel S_{i,n}}{2 n_i} \\
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
\dot{w}_\parallel &= b^z \frac{1}{m_i n_i v_{Ti}} \frac{\partial p_{i\parallel}}{\partial z} - \frac{1}{v_{Ti}} R_\mathrm{CX} n_n \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{1}{v_{Ti}} R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \left(u_{n\parallel} - u_{i\parallel}\right) - \frac{S_{i,\mathrm{mom}}}{m_i n_iv_{Ti}} + \frac{u_{i\parallel} S_{i,n}}{n_i v_{Ti}} \nonumber \\
&\quad + w_\parallel \left( - b^z \frac{\partial u_{i\parallel}}{\partial z} + \frac{b^z}{3 p_i} \frac{\partial q_{i\parallel}}{\partial z} + \frac{b^z}{3 p_i} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \right. \nonumber \\
&\qquad\qquad + \frac{1}{6 T_i} R_\mathrm{CX} n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) \nonumber \\
&\qquad\qquad \left. - \frac{1}{6 p_i} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) - \frac{S_{i,p}}{2 p_i} + \frac{S_{i,n}}{2 n_i} \right) \nonumber \\
&\quad - b^z w_\parallel^2 \frac{\partial v_{Ti}}{\partial z} \\

\dot{w}_\perp &= w_\perp \left( - b^z w_\parallel \frac{\partial v_{Ti}}{\partial z} + \frac{b^z}{3 p_i} \frac{\partial q_{i\parallel}}{\partial z} + \frac{b^z}{3 p_i} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \right . \nonumber \\
&\qquad\qquad + \frac{1}{6 T_i} R_\mathrm{CX} n_n \left(3 T_i + 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) \nonumber \\
&\qquad\qquad \left . - \frac{1}{6 p_i} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) - \frac{S_{i,p}}{2 p_i} + \frac{S_{i,n}}{2 n_i} \right) \nonumber \\

\frac{\dot{F}_i}{F_i} &= 3 b^z w_\parallel \frac{\partial v_{Ti}}{\partial z} - \frac{b^z}{p_i} \frac{\partial q_{i\parallel}}{\partial z} - \frac{b^z}{p_i} p_{i\parallel} \frac{\partial u_{i\parallel}}{\partial z} \nonumber \\
&\quad - \frac{1}{2 T_i} R_\mathrm{CX} n_n \left(3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2\right) \nonumber \\
&\quad + \frac{1}{2 p_i} R_\mathrm{ioniz} n_e n_n \left(3 T_n - 3 T_i + m_i (u_{i\parallel} - u_{n\parallel})^2\right) + \frac{3 S_{i,p}}{2 p_i} - \frac{3 S_{i,n}}{2 n_i} \nonumber \\
&\quad - b^z \frac{v_{Ti} w_\parallel}{n_i} \frac{\partial n}{\partial z} + b^z \frac{\partial u_{i\parallel}}{\partial z} - R_\mathrm{ioniz} \frac{n_e n_n}{n_i} - \frac{S_{i,n}}{n_i} \\
\end{align}
```
However, the expressions for $\dot{w}_\parallel$, $\dot{w}_\perp$, and
$\dot F_i$ become much longer. In the code, we have to calculate
$\partial n_i/\partial t$, etc. anyway to evolve the moment equations, so it
will be simpler to save these values, and implement the kinetic equation
coefficients in terms of $\partial n_i/\partial t$, etc. Especially if/when new
terms (e.g.  inter-species friction) are added to the model, this approach will
minimise the number of places in the code that need to be updated.

Neutral equations
-----------------

Neutrals do not see $\phi$ in their kinetic equation, and signs of ion-neutral
reaction terms are flipped
```math
\begin{align}
& \frac{\partial f_n}{\partial t} + v_\parallel \frac{\partial f_n}{\partial z}
    = \mathcal{C}_{ni}[f_n,f_i] + S_n \\
& \mathcal{C}_{ni}[f_n,f_i] = -\mathcal{C}_{in}[f_i,f_n]
    = R_\mathrm{CX} (n_n f_i - n_i f_n) - R_\mathrm{ioniz} n_e f_n
\end{align}
```
The moment equations are therefore very similar to those of the ions
```math
\begin{align}
& \frac{\partial n_n}{\partial t} + \frac{\partial}{\partial z}\left( n_n u_{n\parallel} \right)
    = - R_\mathrm{ioniz} n_e n_n + S_{n,n} \\

& m_n \frac{\partial}{\partial t}(n_n u_{n\parallel})
  + m_n \frac{\partial}{\partial z}(n_n u_{n\parallel}^2)
  + \frac{\partial p_{n\parallel}}{\partial z} \nonumber \\
&\quad= -R_\mathrm{CX} m_n n_i n_n (u_{n\parallel} - u_{i\parallel})
    - R_\mathrm{ioniz} m_n n_e n_n u_{n\parallel}
    + S_{n,\mathrm{mom}} \\

& \frac{3}{2} \frac{\partial}{\partial t} \left( \frac{3 p_n}{m_n} + n_n u_{n\parallel}^2 \right)
  + \frac{3}{2} \frac{\partial}{\partial z} \left[ \frac{2 q_{n\parallel}}{m_n} + 2\frac{p_{n\parallel}}{m_n} u_{n\parallel} + \frac{3 p_n}{m_n} u_{n\parallel} + n_n u_{n\parallel}^3 \right] \nonumber \\
&\quad= \frac{3}{2} R_\mathrm{CX} \frac{n_i n_n}{m_i} \left(3 T_i + m_i u_{i\parallel}^2 - 3 T_n - m_i u_{n\parallel}^2 \right)
      - \frac{3}{2} R_\mathrm{ioniz} \frac{n_e n_n}{m_i} \left(3 T_n + m_i u_{n\parallel}^2 \right) \nonumber \\
&\qquad+ \frac{3}{m_i} S_{n,E} \\

& \frac{3}{2} \frac{\partial p_n}{\partial t}
  + \frac{\partial q_{n\parallel}}{\partial z} + p_{n\parallel} \frac{\partial u_{n\parallel}}{\partial z}
  + \frac{3}{2} u_{n\parallel} \frac{\partial p_n}{\partial z} + \frac{3}{2} p_n \frac{\partial u_{n\parallel}}{\partial z} \nonumber \\
&\quad= \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
      - \frac{3}{2} R_\mathrm{ioniz} n_e n_n T_n \nonumber \\
&\qquad+ \frac{3}{2} S_{n,p} \\
\end{align}
```
The alternative forms of equation for $\partial u_{n\parallel}\partial t$ and
$\partial v_{Tn} / \partial t$ may also be useful
```math
\begin{align}
&m_n n_n \frac{\partial u_{n\parallel}}{\partial t}
 + m_n n_n u_{n\parallel} \frac{\partial u_{n\parallel}}{\partial z} \nonumber \\
&= - \frac{\partial p_{n\parallel}}{\partial z}
   - R_\mathrm{CX} m_n n_i n_n (u_{n\parallel} - u_{i\parallel})
   + S_{n,\mathrm{mom}} - m_n u_{n\parallel} S_{n,n} \\

& \frac{3}{2} m_n n_n v_{Tn} \left( \frac{\partial v_{Tn}}{\partial t} + u_{n\parallel} \frac{\partial v_{Tn}}{\partial z} \right) \nonumber \\
&\quad= - \frac{\partial q_{n\parallel}}{\partial z} - p_{n\parallel} \frac{\partial u_{n\parallel}}{\partial z}
     \nonumber \\
&\qquad+ \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right) \nonumber \\
&\qquad+ \frac{3}{2} S_{n,p} - \frac{3}{2} T_n S_{n,n} \\
\end{align}
```
The 3 variants on the moment-kinetic equation, given in the following
subsections, are also very similar to the ion ones.

### Separate $n_n$

```math
\begin{align}
& \frac{\partial F_n}{\partial t} + v_\parallel \frac{\partial F_n}{\partial z}
  + \left( \frac{(v_\parallel - u_{n\parallel})}{n_n} \frac{\partial n_n}{\partial z} - \frac{\partial u_{n\parallel}}{\partial z} + \frac{1}{n_n} S_{n,n} \right) F_n \nonumber \\
&\quad= R_\mathrm{CX} n_i (F_i - F_n) + \frac{1}{n_n} S_n \\
\end{align}
```

### Separate $n_n$ and $u_{n\parallel}$

```math
\begin{align}
& \frac{\partial F_n}{\partial t}
  + (\hat{w}_\parallel + u_{n\parallel}) \frac{\partial F_n}{\partial z} \nonumber \\
  &\quad- \left( \hat{w}_\parallel \frac{\partial u_{n\parallel}}{\partial z}
                 - \frac{1}{m_n n_n} \frac{\partial p_{n\parallel}}{\partial z}
                 - R_\mathrm{CX} n_i (u_{n\parallel} - u_{i\parallel})
                 + \frac{1}{m_n n_n} S_{n,\mathrm{mom}}
               \right) \frac{\partial F_n}{\partial \hat{w}_\parallel} \nonumber \\
  &\quad+ \left( \frac{\hat{w}_\parallel}{n_n} \frac{\partial n_n}{\partial z} - \frac{\partial u_{n\parallel}}{\partial z} + \frac{1}{n_n} S_{n,n} \right) F_n \nonumber \\
&\quad= R_\mathrm{CX} n_i (F_i - F_n) + \frac{1}{n_n} S_n \\
\end{align}
```

### Full moment-kinetics (separate $n_n$, $u_{n\parallel}$ and $p_{n}$)

```math
\begin{align}
&\frac{\partial F_n}{\partial t} + \dot{z} \frac{\partial F_n}{\partial z} + \dot{w}_\parallel \frac{\partial F_n}{\partial w_\parallel} + \dot{w}_\perp \frac{\partial F_n}{\partial w_\perp}
    = \dot{F}_n + \mathcal{C}_n + \frac{v_{Tn}^3}{n_n} S_n \\

&\dot{z} = v_{Tn} w_\parallel + u_{n\parallel} \\

&\dot{w}_\parallel = - \left( \frac{1}{v_{Tn}} \frac{\partial u_{n\parallel}}{\partial t} + \left( w_\parallel + \frac{u_{n_\parallel}}{v_{Tn}} \right) \frac{\partial u_{n\parallel}}{\partial z} \right. \nonumber \\
    &\qquad\quad  \left. + \frac{w_\parallel}{v_{Tn}} \frac{\partial v_{Tn}}{\partial t} + w_\parallel \left( w_\parallel + \frac{u_{n\parallel}}{v_{Tn}} \right) \frac{\partial v_{Tn}}{\partial z} \right) \\

&\dot{w}_\perp = -\left( \frac{w_\perp}{v_{Tn}} \frac{\partial v_{Tn}}{\partial t} + \left( w_\parallel + \frac{u_{n\parallel}}{v_{Tn}} \right) w_\perp \frac{\partial v_{Tn}}{\partial z} \right) \\

&\frac{\dot{F}_n}{F_n} = \frac{3}{v_{Tn}} \frac{\partial v_{Tn}}{\partial t} + \frac{3 (v_{Tn} w_\parallel + u_{n\parallel})}{v_{Tn}} \frac{\partial v_{Tn}}{\partial z}
                   - \frac{1}{n_n} \frac{\partial n_n}{\partial t} - \frac{(v_{Tn} w_\parallel + u_{n\parallel})}{n_n} \frac{\partial n_n}{\partial z} \\

\mathcal{C}_n &= R_\mathrm{CX} n_i \left( \frac{v_{Tn}^3}{v_{Ti}^3} F_i - F_n \right) - R_\mathrm{ioniz} n_e F_n \\
\end{align}
```

Again, we could substitute in the moment equations
```math
\begin{align}
\dot{w}_\parallel &= w_\parallel \frac{\partial u_{n\parallel}}{\partial z}
                     + \frac{1}{m_n n_n v_{Tn}} \frac{\partial p_{n\parallel}}{\partial z}
                     + R_\mathrm{CX} \frac{n_i}{v_{Tn}} (u_{n\parallel} - u_{i\parallel})
                     - \frac{1}{m_n n_n v_{Tn}} (S_{n,\mathrm{mom}} - u_{n\parallel} S_{n,n}) \nonumber \\
        &\qquad\quad - w_\parallel^2 \frac{\partial v_{Tn}}{\partial z}
                     + \frac{w_\parallel}{3 p_n} \frac{\partial q_{n\parallel}}{\partial z}
                     + \frac{w_\parallel p_{n\parallel}}{3 p_n} \frac{\partial u_{n\parallel}}{\partial z}
                     - \frac{w_\parallel}{6 p_n} R_\mathrm{CX} n_i n_n \left( 3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
                     - \frac{w_\parallel}{2 p_n} S_{n,p} + \frac{w_\parallel}{2 p_n} T_n S_{n,n} \\
\dot{w}_\perp &= -w_\parallel w_\perp \frac{\partial v_{Tn}}{\partial z}
                 + \frac{w_\perp}{3 p_n} \frac{\partial q_{n\parallel}}{\partial z}
                 + \frac{w_\perp p_{n\parallel}}{3 p_n} \frac{\partial u_{n\parallel}}{\partial z}
                 - \frac{w_\perp}{6 p_n} R_\mathrm{CX} n_i n_n \left( 3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
                 - \frac{w_\perp}{2 p_n} S_{n,p} + \frac{w_\perp}{2 p_n} T_n S_{n,n} \\
\frac{\dot{F}_n}{F_n} &= 3 w_\parallel \frac{\partial v_{Tn}}{\partial z}
                         - \frac{1}{p_n} \frac{\partial q_{n\parallel}}{\partial z}
                         - \frac{p_{n\parallel}}{p_n} \frac{\partial u_{n\parallel}}{\partial z}
                         + \frac{1}{2 p_n} R_\mathrm{CX} n_i n_n \left( 3 T_i - 3 T_n - m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
                         + \frac{3}{2 p_n} S_{n,p} - \frac{5}{2 n_n} S_{n,n} \nonumber \\
            &\qquad\quad - \frac{v_{Tn} w_\parallel}{n_n} \frac{\partial n_n}{\partial z}
                         + \frac{\partial u_{n\parallel}}{\partial z} - R_\mathrm{ioniz} n_e \nonumber \\
\end{align}
```
noting that ionization does not appear in the equations for $\partial
u_{n\parallel} / \partial t$ or $\partial v_{Tn} / \partial t$ so that the only
contributions are from $\partial n_n / \partial t$ that contributes to
$\dot{F}$ and the explicit term in $\mathcal C_n$, and these will cancel.

### Wall boundary condition (Knudsen cosine distribution)

Neutrals returning from the wall belong to a Knudsen cosine distribution,
defined with a wall temperature $T_\mathrm{wall}$, which is given by
[\[Excalibur report TN-05\]](https://excalibur-neptune.github.io/Documents/TN-05_1DDriftKineticWallBoundaryConditions.html)
```math
\begin{align}
f_n(0,v_\parallel>0,v_\perp,t) &= \Gamma_0 f_\mathrm{Kw}(v_\parallel,v_\perp) \\
f_n(L,v_\parallel<0,v_\perp,t) &= \Gamma_L f_\mathrm{Kw}(v_\parallel,v_\perp) \\
\Gamma_0 &= \sum_{s=i,n} 2 \pi \int_{-\infty}^0 dv_\parallel \int_0^\infty dv_\perp v_\perp |v_\parallel| f_s(0,v_\parallel,v_\perp,t) \\
\Gamma_L &= \sum_{s=i,n} 2 \pi \int_0^\infty dv_\parallel \int_0^\infty dv_\perp v_\perp |v_\parallel| f_s(L,v_\parallel,v_\perp,t) \\
f_\mathrm{Kw}(v_\parallel,v_\perp) &= \frac{3}{4\pi} \left( \frac{m_n}{T_w} \right)^2 \frac{|v_\parallel|}{\sqrt{v_\parallel^2 + v_\perp^2}} \exp\left( -\frac{m_n(v_\parallel^2 + v_\perp^2)}{2T_w} \right) \\
\end{align}
```

[Reduction to 2D1V](@id ion_reduction_to_2d1v)
----------------------------------------------

To reduce the model to 2D1V, we take the limit $T_{s\perp} \rightarrow 0$, and
marginalise over $v_\perp$ to remove one velocity space dimension.

One way to do this formally is to assume that
```math
\begin{align}
f_s &= \bar{f}_s(t,z,v_\parallel) f_{s\perp}(v_\perp) \\
\text{with } f_{s\perp}(v_\perp) &= \frac{\exp(-v_\perp^2/v_{Ts\perp}^2)}{\pi v_{Ts\perp}^2} \\
\text{where } \frac{1}{2} m_s v_{Ts\perp}^2 &= T_\perp \\
\text{and similarly } S_s &= \bar{S}_s(t,z,v_\parallel) f_{s\perp}(v_\perp)
\end{align}
```
 $f_{s\perp}$ is defined so that
```math
\begin{align}
\int f_{s\perp} d^2 v_\perp &= 2\pi \int_0^\infty f_{s\perp} v_\perp dv_\perp \nonumber \\
  &= 2\pi \int_0^\infty \frac{\exp(-v_\perp^2/v_{Ts\perp}^2)}{\pi v_{Ts\perp}^2} v_\perp dv_\perp \nonumber \\
  &= 2 \int_0^\infty \exp(-x^2) x dx \nonumber \\
  &= 2 \left[ -\frac{1}{2} \exp(-x^2) \right]_0^\infty \nonumber \\
  &= 1 \\
\int v_\perp f_{s\perp} d^2 v_\perp &= 2\pi \int_0^\infty v_\perp f_{s\perp} v_\perp dv_\perp \nonumber \\
  &= \lim_{v_{Ts\perp}\rightarrow 0} 2\pi \int_0^\infty v_\perp^2 \frac{\exp(-v_\perp^2/v_{Ts\perp}^2)}{\pi v_{Ts\perp}^2} dv_\perp \nonumber \\
  &= \lim_{v_{Ts\perp}\rightarrow 0} 2 v_{Ts\perp} \int_0^\infty x^2 \exp(-x^2) dx \nonumber \\
  &= \lim_{v_{Ts\perp}\rightarrow 0} 2 v_{Ts\perp} \left( \left[ -\frac{1}{2} x \exp(-x^2) \right]_0^\infty + \frac{1}{2} \int_0^\infty \exp(-x^2) \right) \nonumber \\
  &= \lim_{v_{Ts\perp}\rightarrow 0} 2 v_{Ts\perp} \left( 0 + \frac{1}{2} \frac{\sqrt{\pi}}{2} \right) \nonumber \\
  &= 0
\end{align}
```
and so
```math
\begin{align}
\int f_s(t,z,v_\parallel,v_\perp) d^2 v_\perp = \bar{f}_s(t,z,v_\parallel) \\
\int S_s(t,z,v_\parallel,v_\perp) d^2 v_\perp = \bar{S}_s(t,z,v_\parallel) \\
\end{align}
```
Integrals with any higher powers of $v_\perp$ also vanish.

The source integrals are the same, but can be written in terms of $\bar S_s$,
$S_{s,n} = \int \bar S_{s} dv_\parallel$,
$S_{s,\mathrm{mom}} = \int m_s v_\parallel \bar S_{s} dv_\parallel$,
$S_{s,E} = \int \frac{1}{2} m_s v_\parallel^2 \bar S_{s} dv_\parallel$,
and
$S_{s,p} = \frac{2}{3} S_{i,E} - \frac{2}{3} u_{i\parallel} S_{i,\mathrm{mom}} + \frac{1}{3} m_s u_{s\parallel}^2 S_{s,n}$.

The marginalised shape function must be marginalised over $w_\perp$, not
$v_\perp$ because $F_s$ is dimensionless, and $\bar F_s$ should be
dimensionless too.
```math
\begin{align}
\bar{F}_s(t,z,w_\parallel) &= \int F_s(t,z,w_\parallel,w_\perp) d^2 w_\perp \nonumber \\
  &= \int F_s(t,z,w_\parallel,w_\perp) \frac{1}{v_{Ts}^2} d^2 v_\perp \nonumber \\
  &= \int \frac{v_{Ts}^3}{n_s} f_s(t,z,v_\parallel,v_\perp) \frac{1}{v_{Ts}^2} d^2 v_\perp \nonumber \\
  &= \frac{v_{Ts}}{n_s} \int f_s(t,z,v_\parallel,v_\perp) d^2 v_\perp \nonumber \\
  &= \frac{v_{Ts}}{n_s} \bar{f}_s(t,z,v_\parallel) \\
\end{align}
```
and like the integrals above,
$\int w_\perp F_s(t,z,w_\parallel,w_\perp) d^2 w_\perp = 0$.

Setting $T_\perp = 0$ in the expressions in section [Moment kinetic
equations](@ref),
```math
\begin{align}
p_s &= \frac{1}{3}(p_{s\parallel} + 2p_{s\perp}) = \frac{p_{s\parallel}}{3} \\
T_s &= \frac{1}{3}(T_{s\parallel} + 2T_{s\perp}) = \frac{T_{s\parallel}}{3} \\
v_{Ts} &= \sqrt{\frac{2(T_{s\parallel} + 2 T_{s\perp})}{3 m_s}} = \sqrt{\frac{2T_{s\parallel}}{3 m_s}} \\
\end{align}
```
The parallel heat flux reduces to
```math
\begin{align}
q_{s\parallel} &= \int \frac{m_s}{2} \left( (v_\parallel - u_{s\parallel})^2 + v_\perp^2 \right) (v_\parallel - u_{s\parallel}) f_s d^3 v \nonumber \\
    &= \int \frac{m_s}{2} \left( (v_\parallel - u_{s\parallel})^2 \right) (v_\parallel - u_{s\parallel}) \left(\int f_s d^2 v_\perp \right) dv_\parallel \nonumber \\
    &\quad + \cancel{\int \frac{m_s}{2} (v_\parallel - u_{s\parallel}) \left(\int v_\perp^2 f_s d^2 v_\perp \right) d^3 v} \nonumber \\
    &= \int \frac{m_s}{2} (v_\parallel - u_{s\parallel})^3 \bar{f}_s dv_\parallel \nonumber \\
\end{align}
```
or in terms of $\bar F_s$
```math
\begin{align}
q_{s\parallel} &= n_s v_{Ts}^3 \int \frac{m_s}{2} \left( w_\parallel^2 + \cancel{w_\perp^2} \right) w_\parallel F_s d^3 w \nonumber \\
q_{s\parallel} &= n_s v_{Ts}^3 \int \frac{m_s}{2} w_\parallel^3 \left(\int F_s d^2 w_\perp \right) dw_\parallel \\
q_{s\parallel} &= n_s v_{Ts}^3 \int \frac{m_s}{2} w_\parallel^3 \bar{F}_s dw_\parallel \\
\end{align}
```

The moment constraints reduce to
```math
\begin{align}
1 &= \int dw_\parallel \int d^2 w_\perp F_s(t,z,w_\parallel,w_\perp) = \int dw_\parallel \bar{F}_s(t,z,w_\parallel) \\
0 &= \int dw_\parallel \int d^2 w_\perp w_\parallel F_s(t,z,w_\parallel,w_\perp) = \int dw_\parallel w_\parallel \bar{F}_s(t,z,w_\parallel) \\
\frac{3}{2} &= \int dw_\parallel \int d^2 w_\perp (w_\parallel^2 + \cancel{w_\perp^2}) F_s(t,z,w_\parallel,w_\perp) = \int dw_\parallel w_\parallel^2 \bar{F}_s(t,z,w_\parallel,w_\perp) \\
\end{align}
```

### 2D1V ion moment equations

The moment equations are identical to the 2D2V case, although in 2D1V
$p_{i\perp} = 0$ so that $p_i = p_{i\parallel}/3$.

### 2D1V ion kinetic equation

When we marginalise the ion kinetic equation to reduce it to 1D1V form, we note
that $\dot w_\perp \propto w_\perp$, so the term
```math
\begin{align}
\dot{w}_\perp \frac{\partial F_i}{\partial w_\perp} = - \frac{w_\perp}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right) \frac{\partial F_i}{\partial w_\perp}
\end{align}
```
and marginalising
```@raw html
<details>
<summary style="text-align:center">[ intermediate steps ]</summary>
```
```math
\begin{align}
\int w_\perp \frac{\partial F_i}{\partial w_\perp} d^2 w_\perp
    &= 2 \pi \int_0^\infty w_\perp \frac{\partial F_i}{\partial w_\perp} w_\perp dw_\perp \nonumber \\
    &= 2 \pi \int_0^\infty w_\perp^2 \frac{\partial F_i}{\partial w_\perp} dw_\perp \nonumber \\
    &= 2 \pi \left(
                   \left[ w_\perp^2 F_i \right]_0^\infty
                   - \int_0^\infty 2 w_\perp F_i dw_\perp
             \right) \nonumber \\
    &= -2 \pi \int_0^\infty 2 w_\perp F_i dw_\perp \nonumber \\
    &= -2 \pi \int_0^\infty 2 w_\perp \frac{v_{Ti}^3}{n_i} f_i dw_\perp \nonumber \\
    &= -2 \pi \frac{v_{Ti}^3}{n_i} \frac{1}{v_{Ti}^2} \int_0^\infty 2 v_\perp f_i dv_\perp \nonumber \\
    &= -2 \pi \frac{v_{Ti}}{n_i} \bar{f}_i \int_0^\infty 2 v_\perp f_{i\perp} dv_\perp \nonumber \\
    &= -2 \pi \frac{v_{Ti}}{n_i} \bar{f}_i \lim_{v_{Ti\perp} \rightarrow 0} \int_0^\infty 2 v_\perp \frac{\exp(-v_\perp^2/v_{Ti\perp}^2)}{\pi v_{Ti\perp}^2} dv_\perp \nonumber \\
    &= -2 \pi \frac{v_{Ti}}{n_i} \bar{f}_i \lim_{v_{Ti\perp} \rightarrow 0} \int_0^\infty 2 x \frac{\exp(-x^2)}{\pi} dx \nonumber \\
    &= -4 \frac{v_{Ti}}{n_i} \bar{f}_i \lim_{v_{Ti\perp} \rightarrow 0} \left[ -\frac{1}{2} \exp(-x^2) \right]_0^\infty \nonumber \\
    &= -2 \frac{v_{Ti}}{n_i} \bar{f}_i \nonumber \\
\end{align}
```
```@raw html
</details>
```
```math
\begin{align}
\int w_\perp \frac{\partial F_i}{\partial w_\perp} d^2 w_\perp
    &= -2 \bar{F}_i \\
\end{align}
```
This term cancels with the contribution to $\dot F_i$ that was associated with
having three powers of $v_{Ti}$ in
$F_s(t,z,w_\parallel,w_\perp) = \frac{v_{Ti}^3}{n_i} f_s(t,z,v_\parallel,v_\perp)$
rather than the one power of $v_{Ti}$ in
$\bar{F}_s(t,z,w_\parallel,w_\perp) = \frac{v_{Ti}}{n_i} \bar{f}_s(t,z,v_\parallel,v_\perp)$.
Finally
```math
\begin{align}
&\frac{\partial \bar{F}_i}{\partial t} + \dot{z} \frac{\partial \bar{F}_i}{\partial z} + \dot{w}_\parallel \frac{\partial \bar{F}_i}{\partial w_\parallel} + \dot{w}_\perp \frac{\partial \bar{F}_i}{\partial w_\perp}
    = \dot{\bar{F}}_i + \bar{\mathcal{C}}_i + \frac{v_{Ti}}{n_i} \bar{S}_i \\
\end{align}
```
(noting that
$\int S_i d^2 w_\perp = v_{Ti}^{-2} \int S_i d^2 v_\perp = v_{Ti}^{-2} \bar S_i$),
where (for now?) we assume there is no $B$-variation for the 2D1V case, so
$\dot v_\perp = 0$, giving
```math
\begin{align}
\dot{r} &= v_E^r \\

\dot{z} &= v_E^z + b^z v_{Ti} w_\parallel + b^z u_{i\parallel} \\

\dot{w}_\parallel &=
  \frac{\dot{v}_\parallel}{v_{Ti}}
  - \frac{1}{v_{Ti}} \left( \frac{\partial u_{i\parallel}}{\partial t} + \dot{r} \frac{\partial u_{i\parallel}}{\partial r} + \dot{z} \frac{\partial u_{i\parallel}}{\partial z} \right)
  + \frac{w_\parallel}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right) \\

\frac{\dot{\bar{F}}_i}{\bar{F}_i} &= \frac{3}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right)
                   - \frac{1}{n_i} \left( \frac{\partial n_i}{\partial t} + \dot{r} \frac{\partial n_i}{\partial r} + \dot{z} \frac{\partial n_i}{\partial z} \right)
                   - \underbrace{\frac{2}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right)}_\text{term coming from $\dot{w}_\perp \partial F_i / \partial w_\perp$} \nonumber \\
&= \frac{1}{v_{Ti}} \left( \frac{\partial v_{Ti}}{\partial t} + \dot{r} \frac{\partial v_{Ti}}{\partial r} + \dot{z} \frac{\partial v_{Ti}}{\partial z} \right)
                   - \frac{1}{n_i} \left( \frac{\partial n_i}{\partial t} + \dot{r} \frac{\partial n_i}{\partial r} + \dot{z} \frac{\partial n_i}{\partial z} \right) \\

\bar{\mathcal{C}}_i &= \frac{v_{Ti}}{n_i} \int C_{ii}[f_i, f_i] d^2 v_\perp - R_\mathrm{CX} n_n \left( \bar{F}_i - \frac{v_{Ti}}{v_{Tn}} \bar{F}_n \right) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \frac{v_{Ti}}{v_{Tn}} \bar{F}_n \\
\end{align}
```

The separate-$n_i$ and separate-$n_i$,$u_{i\parallel}$ equations are simpler to
marginalise, as they contained no $\partial F_i / \partial w_{i\perp}$ term.
* Separate $n_i$
  ```math
  \begin{align}
  & \frac{\partial \bar{F}_i}{\partial t} + v_E^r \frac{\partial \bar{F}_i}{\partial r} + (v_E^z + b^z v_\parallel) \frac{\partial \bar{F}_i}{\partial z}
    - b^z \frac{e}{m_i} \frac{\partial\phi}{\partial z} \frac{\partial \bar{F}_i}{\partial v_\parallel} \nonumber \\
    &+ \left( \frac{(b^z v_\parallel - b^z u_{i\parallel})}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) \bar{F}_i \nonumber \\
  &\quad= \frac{1}{n_i} \int C_{ii}[n_i F_i, n_i F_i] d^2 v_\perp - R_\mathrm{CX} n_n (\bar{F}_i - \bar{F}_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \bar{F}_n + \frac{1}{n_i} \bar{S}_i \\
  \end{align}
  ```
* Separate $n_i$ and $u_{i\parallel}$, with again $\hat w_\parallel = v_\parallel - u_{s\parallel}$
  ```math
  \begin{align}
  & \frac{\partial \bar{F}_i}{\partial t}
    + v_E^r \frac{\partial \bar{F}_i}{\partial r}
    + (v_E^z + b^z \hat{w}_\parallel + b^z u_{i\parallel}) \frac{\partial \bar{F}_i}{\partial z} \nonumber \\
    &\quad- \left( b^z \hat{w}_\parallel \frac{\partial u_{i\parallel}}{\partial z}
                   - b^z \frac{1}{m_i n_i} \frac{\partial p_{i\parallel}}{\partial z}
                   + R_\mathrm{CX} n_n (u_{n\parallel} - u_{i\parallel})
                   + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} u_{n\parallel}
                   + \frac{1}{m_i n_i} S_{i,\mathrm{mom}}
                 \right) \frac{\partial \bar{F}_i}{\partial \hat{w}_\parallel} \nonumber \\
    &\quad+ \left( b^z \frac{\hat{w}_\parallel}{n_i} \frac{\partial n_i}{\partial z} - b^z \frac{\partial u_{i\parallel}}{\partial z} + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} + \frac{1}{n_i} S_{i,n} \right) \bar{F}_i \nonumber \\
  &\quad= \frac{1}{n_i} \int C_{ii}[n_i F_i, n_i F_i] d^2 v_\perp - R_\mathrm{CX} n_n (\bar{F}_i - \bar{F}_n) + R_\mathrm{ioniz} \frac{n_e n_n}{n_i} \bar{F}_n + \frac{1}{n_i} \bar{S}_i \\
  \end{align}
  ```

### 1D1V neutral equations

Similarly setting $T_{i\perp} = 0$ in the moment equations for neutrals
```math
\begin{align}
\frac{\partial n_n}{\partial t} + \frac{\partial}{\partial z}\left( n_n u_{n\parallel} \right)
    &= - R_\mathrm{ioniz} n_e n_n + S_{n,n} \\
\end{align}
```
```math
\begin{align}
& m_n \frac{\partial}{\partial t}(n_n u_{n\parallel})
  + m_n \frac{\partial}{\partial z}(n_n u_{n\parallel}^2)
  + \frac{\partial p_{n\parallel}}{\partial z} \nonumber \\
&\quad= -R_\mathrm{CX} m_n n_i n_n (u_{n\parallel} - u_{i\parallel})
    - R_\mathrm{ioniz} m_n n_e n_n u_{n\parallel}
    + S_{n,\mathrm{mom}} \\

& \frac{3}{2} \frac{\partial p_n}{\partial t}
  + \frac{\partial q_{n\parallel}}{\partial z} + p_{n\parallel} \frac{\partial u_{n\parallel}}{\partial z}
  + \frac{3}{2} u_{n\parallel} \frac{\partial p_n}{\partial z} + \frac{3}{2} p_n \frac{\partial u_{n\parallel}}{\partial z} \nonumber \\
&\quad= \frac{1}{2} R_\mathrm{CX} n_i n_n \left(3 T_i - 3 T_n + m_i (u_{i\parallel} - u_{n\parallel})^2 \right)
      - \frac{3}{2} R_\mathrm{ioniz} n_e n_n T_n \nonumber \\
&\qquad+ \frac{3}{2} S_{n,p} \\
\end{align}
```
Marginalising gives the 1D1V moment-kinetic neutral equation in a very similar
form as for the ions
```math
\begin{align}
&\frac{\partial \bar{F}_n}{\partial t} + \dot{z} \frac{\partial \bar{F}_n}{\partial z} + \dot{w}_\parallel \frac{\partial \bar{F}_n}{\partial w_\parallel}
    = \dot{\bar{F}}_n + \bar{\mathcal{C}}_n + \frac{v_{Tn}}{n_n} \bar{S}_n \\
\end{align}
```
where
```math
\begin{align}
\dot{z} &= v_{Tn} w_\parallel + u_{n\parallel} \\

\dot{w}_\parallel &= - \left( \frac{1}{v_{Tn}} \frac{\partial u_{n\parallel}}{\partial t} + \left( w_\parallel + \frac{u_{n_\parallel}}{v_{Tn}} \right) \frac{\partial u_{n\parallel}}{\partial z} \right. \nonumber \\
    &\qquad\quad  \left. + \frac{w_\parallel}{v_{Tn}} \frac{\partial v_{Tn}}{\partial t} + w_\parallel \left( w_\parallel + \frac{u_{n\parallel}}{v_{Tn}} \right) \frac{\partial v_{Tn}}{\partial z} \right) \\

\frac{\dot{\bar{F}}_n}{\bar{F}_n} &= \frac{1}{v_{Tn}} \frac{\partial v_{Tn}}{\partial t} + \frac{(v_{Tn} w_\parallel + u_{n\parallel})}{v_{Tn}} \frac{\partial v_{Tn}}{\partial z}
                   - \frac{1}{n_n} \frac{\partial n_n}{\partial t} - \frac{(v_{Tn} w_\parallel + u_{n\parallel})}{n_n} \frac{\partial n_n}{\partial z} \\

\bar{\mathcal{C}}_n &= R_\mathrm{CX} n_i \left( \frac{v_{Tn}}{v_{Ti}} \bar{F}_i - \bar{F}_n \right) - R_\mathrm{ioniz} n_e \bar{F}_n \\
\end{align}
```

* Separate $n_n$
  ```math
  \begin{align}
  & \frac{\partial \bar{F}_n}{\partial t} + v_\parallel \frac{\partial \bar{F}_n}{\partial z}
    + \left( \frac{(v_\parallel - u_{n\parallel})}{n_n} \frac{\partial n_n}{\partial z} - \frac{\partial u_{n\parallel}}{\partial z} + \frac{1}{n_n} S_{n,n} \right) \bar{F}_n \nonumber \\
  &\quad= R_\mathrm{CX} n_i (\bar{F}_i - \bar{F}_n) + \frac{1}{n_n} \bar{S}_n \\
  \end{align}
  ```
* Separate $n_n$ and $u_{n\parallel}$
  ```math
  \begin{align}
  & \frac{\partial \bar{F}_n}{\partial t}
    + (\hat{w}_\parallel + u_{n\parallel}) \frac{\partial \bar{F}_n}{\partial z} \nonumber \\
    &\quad- \left( \hat{w}_\parallel \frac{\partial u_{n\parallel}}{\partial z} 
                   - \frac{1}{m_n n_n} \frac{\partial p_{n\parallel}}{\partial z}
                   - R_\mathrm{CX} n_i (u_{n\parallel} - u_{i\parallel})
                   + \frac{1}{m_n n_n} S_{n,\mathrm{mom}}
                 \right) \frac{\partial \bar{F}_n}{\partial \hat{w}_\parallel} \nonumber \\
    &\quad+ \left( \frac{\hat{w}_\parallel}{n_n} \frac{\partial n_n}{\partial z} - \frac{\partial u_{n\parallel}}{\partial z} + \frac{1}{n_n} S_{n,n} \right) \bar{F}_n \nonumber \\
  &\quad= R_\mathrm{CX} n_i (\bar{F}_i - \bar{F}_n) + \frac{1}{n_n} \bar{S}_n \\
  \end{align}
  ```

### Wall boundary condition (Knudsen cosine distribution)

For the 1D1V model, the Knudsen cosine distribution is marginalised over
$v_\perp$
[\[Excalibur report TN-08\]](https://excalibur-neptune.github.io/Documents/TN-08_Numerical11DDriftKineticModelWallBoundaryConditions.html)
```math
\begin{align}
\bar{f}_\mathrm{Kw}(v_\parallel) &= 2 \pi \int_0^\infty dv_\perp v_\perp f_\mathrm{Kw}(v_\parallel,v_\perp) \nonumber \\
    &= 3 \sqrt{\pi} \left( \frac{m_n}{2T_w} \right)^{3/2} |v_\parallel| \,\mathrm{erfc}\!\left( \sqrt{\frac{m_n}{2 T_w}} |v_\parallel| \right) \\
\end{align}
```

Conversion to conventions of 1D1V Excalibur reports
---------------------------------------------------

The notes above follow the conventions of the nD2V systems of equations in the
Excalibur reports.
The original version of the 1D1V system as defined in the Excalibur reports had
some conventions chosen differently. Quantities as defined in that original
version are denoted by $\check{\cdot}$ where they differ from those defined here.

The thermal speed was defined using the parallel temperature
```math
\begin{align}
\check{v}_{Ts} &= \sqrt{\frac{2 T_{s\parallel}}{m_s}} \nonumber \\
               &= \sqrt{\frac{2 (3 T_{s})}{m_s}} \nonumber \\
               &= \sqrt{3} v_{Ts}
\end{align}
```
with a corresponding change in the normalised velocity coordinate
```math
\begin{align}
\check{w}_\parallel &= \frac{v_\parallel - u_{s\parallel}}{\check{v}_{Ts}} \nonumber \\
                    &= \frac{v_\parallel - u_{s\parallel}}{\sqrt{3} v_{Ts}} \nonumber \\
                    &= \frac{w_\parallel}{\sqrt{3}}
\end{align}
```
and in the shape function $\check F_s$ (it might be more consistent for the
symbol for this shape function to have a $\bar{\cdot}$ as well as a
$\check{\cdot}$, but that would look messy so the $\bar{\cdot}$ is omitted)
```math
\begin{align}
\check{F}_s(t,z,\check{w}_\parallel) &= \frac{\check{v}_{Ts}}{n_s} \bar{f}_s(t,z,v_\parallel) \nonumber \\
                                           &= \sqrt{3} \frac{v_{Ts}}{n_s} \bar{f}_s(t,z,v_\parallel) \nonumber \\
                                           &= \sqrt{3} \bar{F}_s(t,z,w_\parallel) \nonumber \\
\end{align}
```
Finally, the parallel heat flux was defined without the factor of $1/2$
```math
\begin{align}
\check{q}_{s\parallel} &= \int m_s (v_\parallel - u_{s\parallel})^3 \bar{f}_s dv_\parallel \nonumber \\
                       &= 2 q_{s\parallel}
\end{align}
```

Note that the different definition of $\check w_\parallel$ compared to
$w_\parallel$ and of $\check F_s$ compared to $\bar F_s$ can change form of the
moment constraints - actually this only affects the energy constraint
```math
\begin{align}
1 &= \int \bar{F}_s dw_\parallel = \int \frac{1}{\sqrt{3}} \check{F}_s \sqrt{3} d\check{w}_\parallel = \int \check{F}_s d\check{w}_\parallel \\
0 &= \int w_\parallel \bar{F}_s dw_\parallel = \int \sqrt{3} \check{w}_\parallel \frac{1}{\sqrt{3}} \check{F}_s \sqrt{3} d\check{w}_\parallel = \sqrt{3} \int \check{w}_\parallel \check{F}_s d\check{w}_\parallel \\
\frac{3}{2} &= \int w_\parallel^2 \bar{F}_s dw_\parallel = \int 3 \check{w}_\parallel^2 \frac{1}{\sqrt{3}} \check{F}_s \sqrt{3} d\check{w}_\parallel = 3 \int \check{w}_\parallel^2 \check{F}_s d\check{w}_\parallel \nonumber \\
\Rightarrow \frac{1}{2} &= \int \check{w}_\parallel^2 \check{F}_s d\check{w}_\parallel \\
\end{align}
```

These conversions translate the equations as given above into the 1D1V forms in
the Excalibur reports.

Dimensionless equations for code
--------------------------------

To put the equations on a computer, we need to make them dimensionless. We
choose a reference density $n_\mathrm{ref}$, temperature $T_\mathrm{ref}$,
length $L_\mathrm{ref}$, and magnetic field $B_\mathrm{ref}$, and use the ion
mass $m_i$ as the reference value (it might be useful to change this to 1 amu
in future when we fully implement multi-species?). Denote dimensionless
variables with a $\hat{\cdot}$ in this section.

We also define a reference speed, derived from the temperature and mass
$c_\mathrm{ref} = \sqrt{T_\mathrm{ref}/m_i}$, chosen so that the relation
```math
\begin{align}
\frac{1}{2} m_i v_{Ts}^2 = T_s
\end{align}
```
de-dimensionalises to
```math
\begin{align}
\frac{1}{2} \hat{v}_{Ts}^2 = \hat{T}_s
\end{align}
```

The full set of dimensionless variables are related to the dimensional ones by
```math
\begin{align}
\hat{n}_s &= \frac{n_s}{n_\mathrm{ref}} \\
\hat{T}_s &= \frac{T_s}{T_\mathrm{ref}} \\
\hat{L}_z &= \frac{L_z}{L_\mathrm{ref}} \\
\hat{B} &= \frac{B}{B_\mathrm{ref}} \\
\hat{m}_i &= \frac{m_i}{m_i} = 1 \\
\hat{m}_e &= \frac{m_e}{m_i} \\
\hat{v}_{Ts} &= \frac{v_{Ts}}{c_\mathrm{ref}} \\
\hat{f}_s &= \frac{c_\mathrm{ref}^3}{n_\mathrm{ref}} f_s \\
\hat{F}_s &= F_s \\
\hat{\bar{f}}_s &= \frac{c_\mathrm{ref}}{n_\mathrm{ref}} \bar{f}_s \\
\hat{\bar{F}}_s &= \bar{F}_s \\
\hat{\phi} &= \frac{e \phi}{T_\mathrm{ref}} \\
\hat{E}_\parallel &= \frac{\partial \hat{\phi}}{\partial \hat z} = \frac{e L_\mathrm{ref}}{T_\mathrm{ref}} E_\parallel \\
\hat{t} &= \frac{c_\mathrm{ref} t}{L_\mathrm{ref}} \\
\frac{\partial}{\partial \hat{t}} &= \frac{L_\mathrm{ref}}{c_\mathrm{ref}} \frac{\partial}{\partial t} \\
\hat{z} &= \frac{z}{L_\mathrm{ref}} \\
\frac{\partial}{\partial \hat{z}} &= L_\mathrm{ref} \frac{\partial}{\partial z} \\
\hat{v}_\parallel &= \frac{v_\parallel}{c_\mathrm{ref}} \\
\frac{\partial}{\partial \hat{v}_\parallel} &= c_\mathrm{ref} \frac{\partial}{\partial v_\parallel} \\
\hat{v}_\perp &= \frac{v_\perp}{c_\mathrm{ref}} \\
\frac{\partial}{\partial \hat{v}_\perp} &= c_\mathrm{ref} \frac{\partial}{\partial v_\perp} \\
\hat{p}_s &= \hat{n}_s \hat{T_s} = \frac{p_s}{n_\mathrm{ref} T_\mathrm{ref}} \\
\hat{q}_{s\parallel} &= \frac{q_{s\parallel}}{m_i n_\mathrm{ref} c_\mathrm{ref}^3} = \frac{q_{s\parallel}}{n_\mathrm{ref} T_\mathrm{ref} c_\mathrm{ref}} \\
\hat{R}_\mathrm{CX} &= \frac{L_\mathrm{ref} n_\mathrm{ref} R_\mathrm{CX}}{c_\mathrm{ref}} \\
\hat{R}_\mathrm{ioniz} &= \frac{L_\mathrm{ref} n_\mathrm{ref} R_\mathrm{ioniz}}{c_\mathrm{ref}} \\
\hat{S}_s &= \frac{L_\mathrm{ref} c_\mathrm{ref}^2 S_s}{n_\mathrm{ref}} \\
\hat{\bar{S}}_s &= \frac{L_\mathrm{ref} \bar{S}_s}{n_\mathrm{ref}} \\
\hat{S}_{s,n} &= \frac{L_\mathrm{ref} S_{s,n}}{c_\mathrm{ref} n_\mathrm{ref}} \\
\hat{S}_{s,\mathrm{mom}} &= \frac{L_\mathrm{ref} S_{s,\mathrm{mom}}}{m_\mathrm{i} n_\mathrm{ref} c_\mathrm{ref}^2} \\
\hat{S}_{s,E} &= \frac{L_\mathrm{ref} S_{s,E}}{c_\mathrm{ref} m_\mathrm{i} n_\mathrm{ref} T_\mathrm{ref}} \\
\hat{S}_{s,p} &= \frac{L_\mathrm{ref} S_{s,p}}{c_\mathrm{ref} m_\mathrm{i} n_\mathrm{ref} T_\mathrm{ref}} \\
\hat{C}_{ii}(\hat{f_i}, \hat{f_i}) &= \frac{L_\mathrm{ref} c_\mathrm{ref}^2}{n_\mathrm{ref}} C_{ii}(f_i, f_i) \\
\hat{f}_\mathrm{Kw} &= c_\mathrm{ref}^4 f_\mathrm{Kw} \\
\hat{\bar{f}}_\mathrm{Kw} &= c_\mathrm{ref}^2 \bar{f}_\mathrm{Kw} \\
\end{align}
```

Using these definitions, the dimensionful equations above are converted to
dimensionless equations mostly just by putting a $\hat{\cdot}$ on all
dimensionful quantities. The exceptions are:
```math
\begin{align}
\hat{\dot{v}}_\parallel
&=\frac{t_\mathrm{ref}}{c_\mathrm{ref}} \dot{v}_\parallel
 = \frac{L_\mathrm{ref}}{c_\mathrm{ref}^2} \dot{v}_\parallel \nonumber \\
&= - \frac{L_\mathrm{ref}}{c_\mathrm{ref}^2} b^z \frac{e}{m_i} \frac{\partial \phi}{\partial z}
 = - \frac{\cancel{L_\mathrm{ref}}}{c_\mathrm{ref}^2} b^z \frac{\cancel{e}}{m_i} \frac{T_\mathrm{ref}}{\cancel{e} \cancel{L_\mathrm{ref}}} \frac{\partial \hat{\phi}}{\partial \hat{z}}
 = - \frac{1}{c_\mathrm{ref}^2} b^z \underbrace{\frac{T_\mathrm{ref}}{m_i}}_{c_\mathrm{ref}^2} \frac{\partial \hat{\phi}}{\partial \hat{z}}
 = - b^z \frac{\partial \hat{\phi}}{\partial \hat{z}} \\

\hat{v}_E^r
&=\frac{v_E^r}{c_\mathrm{ref}}
 = - \frac{1}{c_\mathrm{ref}} \frac{b_\zeta}{B} \frac{\partial \phi}{\partial z}
 = - \frac{1}{c_\mathrm{ref}} \frac{T_\mathrm{ref}}{e B_\mathrm{ref} L_\mathrm{ref}} \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{z}}
 = - \frac{m_i c_\mathrm{ref}}{e B_\mathrm{ref} L_\mathrm{ref}} \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{z}}
 = - \rho_* \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{z}} \\

\hat{v}_E^z
&=\frac{v_E^z}{c_\mathrm{ref}}
 = \frac{1}{c_\mathrm{ref}} \frac{b_\zeta}{B} \frac{\partial \phi}{\partial r}
 = \frac{1}{c_\mathrm{ref}} \frac{T_\mathrm{ref}}{e B_\mathrm{ref} L_\mathrm{ref}} \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{r}}
 = \frac{m_i c_\mathrm{ref}}{e B_\mathrm{ref} L_\mathrm{ref}} \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{r}}
 = \rho_* \frac{b_\zeta}{\hat{B}} \frac{\partial \hat{\phi}}{\partial \hat{r}} \\
\end{align}
```
defining
```math
\begin{align}
\Omega_\mathrm{ref} &= \frac{e B_\mathrm{ref}}{m_i} \\

\rho_* &= \frac{c_\mathrm{ref}}{\Omega_\mathrm{ref} L_\mathrm{ref}}
        = \frac{m_i c_\mathrm{ref}}{e B_\mathrm{ref} L_\mathrm{ref}} \\
\end{align}
```

### Definitions for any species

```math
\begin{align}
\hat{p}_{s\parallel} &= 2\pi\int_{-\infty}^\infty d\hat{v}_\parallel \int_0^\infty d\hat{v}_\perp \hat{v}_\perp
    \hat{m}_s \left( \hat{v}_\parallel - \hat{u}_{s\parallel} \right)^2 \hat{f}_s \nonumber \\
    &= \hat{m}_s \hat{n}_s \hat{v}_{Ts}^2 2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp
        w_\parallel^2 F_s \\

\hat{p}_{s\perp} &= 2\pi\int_{-\infty}^\infty d\hat{v}_\parallel \int_0^\infty d\hat{v}_\perp \hat{v}_\perp
    \hat{m}_s \frac{\hat{v}_\perp^2}{2} \hat{f}_s \\
    &= \frac{1}{2} \hat{m}_s \hat{n}_s \hat{v}_{Ts}^2 2\pi \int_{-\infty}^\infty dw_\parallel \int_0^\infty dw_\perp w_\perp
        w_\perp^2 F_s \\

\hat{p}_s &= \hat{n}_s \hat{T}_s = \frac{\hat{m}_s}{2} \hat{n}_s \hat{v}_{Ts}^2 = \frac{1}{3}(\hat{p}_{s\parallel} + 2\hat{p}_{s\perp}) \\

\hat{T}_s &= \frac{1}{3}(\hat{T}_{s\parallel} + 2\hat{T}_{s\perp}) \\

\hat{v}_{Ts} &= \sqrt{\frac{2\hat{T}_s}{\hat{m}_s}}= \sqrt{\frac{2(\hat{T}_{s\parallel} + 2\hat{T}_{s\perp})}{3\hat{m}_s}} \\

\hat{q}_{s\parallel} &= \int \frac{\hat{m}_s}{2} \left( (\hat{v}_\parallel - \hat{u}_{s\parallel})^2 + \hat{v}_\perp^2 \right) (\hat{v}_\parallel - \hat{u}_{s\parallel}) \hat{f}_s d^3 \hat{v} \\
    &= \hat{n}_s \hat{v}_{Ts}^3 \int \frac{\hat{m}_s}{2} \left( w_\parallel^2 + w_\perp^2 \right) w_\parallel F_s d^3 w \\
\end{align}
```

### Dimensionless 1D2V ion equations

For convenience in future extension to multi-ion-species in future, we keep
$\hat m_i$ in the equations below, even though it is equal to 1 in the
conventions here.

```math
\begin{align}
& \frac{\partial \hat{n}_i}{\partial \hat{t}}
+ \hat{v}_E^r \frac{\partial \hat{n}_i}{\partial \hat{r}}
+ \hat{v}_E^z \frac{\partial \hat{n}_i}{\partial \hat{z}}
+ b^z \frac{\partial}{\partial \hat{z}}\left( \hat{n}_i \hat{u}_{i\parallel} \right)
    = \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n + \hat{S}_{i,n} \\

& \hat{m}_i \frac{\partial}{\partial \hat{t}}(\hat{n}_i \hat{u}_{i\parallel})
  + \hat{m}_i \hat{v}_E^r \frac{\partial}{\partial \hat{r}} (\hat{n}_i \hat{u}_{i\parallel})
  + \hat{m}_i \hat{v}_E^z \frac{\partial}{\partial \hat{z}} (\hat{n}_i \hat{u}_{i\parallel})
  + \hat{m}_i b^z \frac{\partial}{\partial \hat{z}}(\hat{n}_i \hat{u}_{i\parallel}^2)
  + b^z \frac{\partial \hat{p}_{i\parallel}}{\partial \hat{z}}
  + b^z \hat{n}_i \frac{\partial \hat{\phi}}{\partial \hat{z}} \nonumber \\
&\quad= \hat{m}_i \hat{R}_\mathrm{CX} \hat{n}_i \hat{n}_n (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
        + \hat{m}_i \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{u}_{n\parallel}
        + \hat{S}_{i,\mathrm{mom}} \\

& \frac{3}{2} \frac{\partial \hat{p}_i}{\partial \hat{t}}
  + \frac{3}{2} \hat{v}_E^r \frac{\partial \hat{p}_i}{\partial \hat{r}}
  + \frac{3}{2} \hat{v}_E^z \frac{\partial \hat{p}_i}{\partial \hat{z}}
  + b^z \frac{\partial \hat{q}_{i\parallel}}{\partial \hat{z}} + b^z \hat{p}_{i\parallel} \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}}
  + \frac{3}{2} b^z \hat{u}_{i\parallel} \frac{\partial \hat{p}_i}{\partial \hat{z}} + \frac{3}{2} b^z \hat{p}_i \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= - \frac{1}{2} \hat{R}_\mathrm{CX} \hat{n}_i \hat{n}_n \left(3 \hat{T}_i - 3 \hat{T}_n - \hat{m}_i (\hat{u}_{i\parallel} - \hat{u}_{n\parallel})^2 \right)
      + \frac{1}{2} \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \left(3 \hat{T}_n + \hat{m}_i (\hat{u}_{i\parallel} - \hat{u}_{n\parallel})^2 \right) \nonumber \\
&\qquad+ \frac{3}{2} \hat{S}_{i,p} \\
\end{align}
```

#### Full kinetic equation

```math
\begin{align}
\frac{\partial \hat{f}_i}{\partial \hat{t}}
    + \hat{v}_E^r \frac{\partial \hat{f}_i}{\partial \hat{r}}
    + (\hat{v}_E^z + b^z \hat{v}_\parallel) \frac{\partial \hat{f}_i}{\partial \hat{z}}
    - b^z \frac{1}{\hat{m}_i} \frac{\partial\hat{\phi}}{\partial \hat{z}} \frac{\partial \hat{f}_i}{\partial \hat{v}_\parallel}
    = \hat{C}_{ii}[\hat{f}_i, \hat{f}_i] - \hat{R}_\mathrm{CX} (\hat{n}_n \hat{f}_i - \hat{n}_i \hat{f}_n) + \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{f}_n + \hat{S}_i
\end{align}
```

#### Separate $\hat n_i$

In this subsection $\hat F_s = c_\mathrm{ref}^3 f_s$.

```math
\begin{align}
& \frac{\partial \hat{F}_i}{\partial \hat{t}} + \hat{v}_E^r \frac{\partial \hat{F}_i}{\partial \hat{r}} + (\hat{v}_E^z + b^z \hat{v}_\parallel) \frac{\partial \hat{F}_i}{\partial \hat{z}}
  - b^z \frac{1}{\hat{m}_i} \frac{\partial\hat{\phi}}{\partial \hat{z}} \frac{\partial \hat{F}_i}{\partial \hat{v}_\parallel}
  + \left( \frac{(b^z \hat{v}_\parallel - b^z \hat{u}_{i\parallel})}{\hat{n}_i} \frac{\partial \hat{n}_i}{\partial \hat{z}} - b^z \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} + \frac{1}{\hat{n}_i} \hat{S}_{i,n} \right) \hat{F}_i \nonumber \\
&\quad= \frac{1}{\hat{n}_i} \hat{C}_{ii}[\hat{n}_i \hat{F}_i, \hat{n}_i \hat{F}_i] - \hat{R}_\mathrm{CX} \hat{n}_n (\hat{F}_i - \hat{F}_n) + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{F}_n + \frac{1}{\hat{n}_i} \hat{S}_i \\
\end{align}
```

#### Separate $\hat n_i$ and $\hat u_{i\parallel}$

In this subsection $\hat F_s = c_\mathrm{ref}^3 f_s$ and
$\hat{\hat{w}}_\parallel = \hat{w}_\parallel / c_\mathrm{ref}$.

```math
\begin{align}
& \frac{\partial F_i}{\partial t}
  + \hat{v}_E^r \frac{\partial \hat{F}_i}{\partial \hat{r}}
  + (\hat{v}_E^z + b^z \hat{\hat{w}}_\parallel + b^z \hat{u}_{i\parallel}) \frac{\partial \hat{F}_i}{\partial \hat{z}} \nonumber \\
  &\quad- \left( b^z \hat{\hat{w}}_\parallel \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}}
                 - b^z \frac{1}{\hat{m}_i \hat{n}_i} \frac{\partial \hat{p}_{i\parallel}}{\partial \hat{z}}
                 + \hat{R}_\mathrm{CX} \hat{n}_n (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
                 + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{u}_{n\parallel}
                 + \frac{1}{\hat{m}_i \hat{n}_i} \hat{S}_{i,\mathrm{mom}}
               \right) \frac{\partial \hat{F}_i}{\partial \hat{\hat{w}}_\parallel} \nonumber \\
  &\quad+ \left( b^z \frac{\hat{\hat{w}}_\parallel}{\hat{n}_i} \frac{\partial \hat{n}_i}{\partial \hat{z}} - b^z \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} + \frac{1}{\hat{n}_i} \hat{S}_{i,n} \right) \hat{F}_i \nonumber \\
&\quad= \frac{1}{\hat{n}_i} \hat{C}_{ii}[\hat{n}_i \hat{F}_i, \hat{n}_i \hat{F}_i] - \hat{R}_\mathrm{CX} \hat{n}_n (\hat{F}_i - \hat{F}_n) + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{F}_n + \frac{1}{\hat{n}_i} \hat{S}_i \\
\end{align}
```

#### Full moment-kinetics (separate $\hat n_i$, $\hat u_{i\parallel}$ and $\hat p_i$)

Note that we do not put a $\hat{\cdot}$ on the already dimensionless quantities
$F_s$, $w_\parallel$, $w_\perp$.

```math
\begin{align}
&\frac{\partial F_i}{\partial \hat{t}} + \hat{\dot{r}} \frac{\partial F_i}{\partial \hat{r}} + \hat{\dot{z}} \frac{\partial F_i}{\partial \hat{z}} + \hat{\dot{w}}_\parallel \frac{\partial F_i}{\partial w_\parallel} + \hat{\dot{w}}_\perp \frac{\partial F_i}{\partial w_\perp}
    = \hat{\dot{F}}_i + \hat{\mathcal{C}}_i + \frac{\hat{v}_{Ti}^3}{\hat{n}_i} \hat{S}_i \\

&\hat{\dot{r}} = \hat{v}_E^r \\

&\hat{\dot{z}} = \hat{v}_E^z + b^z \hat{v}_{Ti} w_\parallel + b^z \hat{u}_{i\parallel} \\

&\hat{\dot{w}}_\parallel =
  \frac{\hat{\dot{v}}_{i\parallel}}{\hat{v}_{Ti}}
  - \frac{1}{\hat{v}_{Ti}} \left( \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{t}} + \hat{\dot{r}} \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{r}} + \hat{\dot{z}} \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} \right)
  - \frac{w_\parallel}{\hat{v}_{Ti}} \left( \frac{\partial \hat{v}_{Ti}}{\partial \hat{t}} + \hat{\dot{r}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{r}} + \hat{\dot{z}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{z}} \right) \\

&\hat{\dot{v}}_{i\parallel} = - b^z \frac{1}{\hat{m}_i} \frac{\partial \hat{\phi}}{\partial \hat{z}} \\

&\hat{\dot{w}}_\perp = \hat{\dot{v}}_{i\perp} - \frac{w_\perp}{\hat{v}_{Ti}} \left( \frac{\partial \hat{v}_{Ti}}{\partial \hat{t}} + \hat{\dot{r}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{r}} + \hat{\dot{z}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{z}} \right) \\

&\hat{\dot{v}}_{i\perp} = 0 \\

&\frac{\hat{\dot{F}}_i}{F_i} =
  \frac{3}{\hat{v}_{Ti}} \left( \frac{\partial \hat{v}_{Ti}}{\partial \hat{t}}
                                + \hat{\dot{r}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{r}}
                                + \hat{\dot{z}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{z}} \right)
  - \frac{1}{\hat{n}_i} \left( \frac{\partial \hat{n}_i}{\partial \hat{t}}
                               + \hat{\dot{r}} \frac{\partial \hat{n}_i}{\partial \hat{r}}
                               + \hat{\dot{z}} \frac{\partial \hat{n}_i}{\partial \hat{z}} \right) \\

&\hat{\mathcal{C}}_i = \frac{\hat{v}_{Ti}^3}{\hat{n}_i} \hat{C}_{ii}[\frac{\hat{n}_i F_i}{\hat{v}_{Ti}^3}, \frac{\hat{n}_i F_i}{\hat{v}_{Ti}^3}] - \hat{R}_\mathrm{CX} \hat{n}_n \left( F_i - \frac{\hat{v}_{Ti}^3}{\hat{v}_{Tn}^3} F_n \right) + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \frac{\hat{v}_{Ti}^3}{\hat{v}_{Tn}^3} \hat{F}_n \\
\end{align}
```

#### Collision coefficient

The coefficient in front of the Fokker-Planck collision operator is
```math
\begin{align}
\gamma_{ss'} = \frac{2 \pi Z_s^2 Z_{s'}^2 e^4 \log\Lambda_{ss'}}{(4 \pi \epsilon_0)^2}
\end{align}
```
and is made dimensionless as
```math
\begin{align}
\hat{\gamma}_{ss'} = \frac{n_\mathrm{ref} t_\mathrm{ref}}{m_\mathrm{ref}^2 c_\mathrm{ref}^3} \gamma_{ss'}
                   = \frac{n_\mathrm{ref} L_\mathrm{ref}}{m_\mathrm{ref}^2 c_\mathrm{ref}^4} \gamma_{ss'}
                   = \frac{1}{2} \frac{n_\mathrm{ref} Z_s^2 Z_{s'}^2 e^4 \log\Lambda_{ss'} L_\mathrm{ref}}{4 \pi \epsilon_0^2 m_\mathrm{ref}^2 c_\mathrm{ref}^4}
\end{align}
```
 $\hat \gamma_{ss'}$ is called `nuii` in [`moment_kinetics.fokker_planck`](@ref).

`nuii` can also be set manually in the `[fokker_planck_collisions]`
input section, which could be thought of as choosing $\log\Lambda_{ss'}$ to set
the requested dimensionless `nuii`.

### Dimensionless 1D2V neutral equations

```math
\begin{align}
& \frac{\partial \hat{n}_n}{\partial \hat{t}} + \frac{\partial}{\partial \hat{z}}\left( \hat{n}_n \hat{u}_{n\parallel} \right)
    = - \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n + \hat{S}_{n,n} \\

& \hat{m}_n \frac{\partial}{\partial \hat{t}}(\hat{n}_n \hat{u}_{n\parallel})
  + \hat{m}_n \frac{\partial}{\partial \hat{z}}(\hat{n}_n \hat{u}_{n\parallel}^2)
  + \frac{\partial \hat{p}_{n\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{R}_\mathrm{CX} \hat{m}_n \hat{n}_i \hat{n}_n (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
    - \hat{R}_\mathrm{ioniz} \hat{m}_n \hat{n}_e \hat{n}_n \hat{u}_{n\parallel}
    + \hat{S}_{n,\mathrm{mom}} \\

& \frac{3}{2} \frac{\partial \hat{p}_n}{\partial \hat{t}}
  + \frac{\partial \hat{q}_{n\parallel}}{\partial \hat{z}} + \hat{p}_{n\parallel} \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}}
  + \frac{3}{2} \hat{u}_{n\parallel} \frac{\partial \hat{p}_n}{\partial \hat{z}} + \frac{3}{2} \hat{p}_n \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= \frac{1}{2} \hat{R}_\mathrm{CX} \hat{n}_i \hat{n}_n \left(3 \hat{T}_i - 3 \hat{T}_n + \hat{m}_i (\hat{u}_{i\parallel} - \hat{u}_{n\parallel})^2 \right)
      - \frac{3}{2} \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{T}_n \nonumber \\
&\qquad+ \frac{3}{2} \hat{S}_{n,p} \\
\end{align}
```

#### Full kinetic equation

```math
\begin{align}
& \frac{\partial \hat{f}_n}{\partial \hat{t}} + \hat{v}_\parallel \frac{\partial \hat{f}_n}{\partial \hat{t}}
    = \hat{R}_\mathrm{CX} (\hat{n}_n \hat{f}_i - \hat{n}_i \hat{f}_n) - \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{f}_n + \hat{S}_n \\
\end{align}
```

#### Separate $\hat n_n$

In this subsection $\hat F_s = c_\mathrm{ref}^3 f_s$.

```math
\begin{align}
& \frac{\partial \hat{F}_n}{\partial \hat{t}} + \hat{v}_\parallel \frac{\partial \hat{F}_n}{\partial \hat{z}}
  + \left( \frac{(\hat{v}_\parallel - \hat{u}_{n\parallel})}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} - \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} + \frac{1}{\hat{n}_n} \hat{S}_{n,n} \right) \hat{F}_n \nonumber \\
&\quad= \hat{R}_\mathrm{CX} \hat{n}_i (\hat{F}_i - \hat{F}_n) + \frac{1}{\hat{n}_n} \hat{S}_n \\
\end{align}
```

#### Separate $\hat n_n$ and $\hat u_{n\parallel}$

In this subsection $\hat F_s = c_\mathrm{ref}^3 f_s$ and
$\hat{\hat{w}}_\parallel = \hat{w}_\parallel / c_\mathrm{ref}$.

```math
\begin{align}
& \frac{\partial \hat{F}_n}{\partial \hat{t}}
  + (\hat{\hat{w}}_\parallel + \hat{u}_{n\parallel}) \frac{\partial \hat{F}_n}{\partial \hat{z}} \nonumber \\
  &\quad- \left( \hat{\hat{w}}_\parallel \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}}
                 - \frac{1}{\hat{m}_n \hat{n}_n} \frac{\partial \hat{p}_{n\parallel}}{\partial \hat{z}}
                 - \hat{R}_\mathrm{CX} \hat{n}_i (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
                 + \frac{1}{\hat{m}_n \hat{n}_n} \hat{S}_{n,\mathrm{mom}}
               \right) \frac{\partial \hat{F}_n}{\partial \hat{\hat{w}}_\parallel} \nonumber \\
  &\quad+ \left( \frac{\hat{\hat{w}}_\parallel}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} - \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} + \frac{1}{\hat{n}_n} \hat{S}_{n,n} \right) \hat{F}_n \nonumber \\
&\quad= \hat{R}_\mathrm{CX} \hat{n}_i (\hat{F}_i - \hat{F}_n) + \frac{1}{\hat{n}_n} \hat{S}_n \\
\end{align}
```

#### Full moment-kinetics (separate $\hat n_n$, $\hat u_{n\parallel}$ and $\hat p_{n}$)

Note that we do not put a $\hat{\cdot}$ on the already dimensionless quantities
$F_s$, $w_\parallel$, $w_\perp$.

```math
\begin{align}
&\frac{\partial F_n}{\partial \hat{t}} + \hat{\dot{z}} \frac{\partial F_n}{\partial \hat{z}} + \hat{\dot{w}}_\parallel \frac{\partial F_n}{\partial w_\parallel} + \hat{\dot{w}}_\perp \frac{\partial F_n}{\partial w_\perp}
    = \hat{\dot{F}}_n + \hat{\mathcal{C}}_n + \frac{\hat{v}_{Tn}^3}{\hat{n}_n} \hat{S}_n \\

&\hat{\dot{z}} = \hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel} \\

&\hat{\dot{w}}_\parallel = - \left( \frac{1}{\hat{v}_{Tn}} \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{t}} + \left( w_\parallel + \frac{\hat{u}_{n_\parallel}}{\hat{v}_{Tn}} \right) \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} \right. \nonumber \\
    &\qquad\quad  \left. + \frac{w_\parallel}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{t}} + w_\parallel \left( w_\parallel + \frac{\hat{u}_{n\parallel}}{\hat{v}_{Tn}} \right) \frac{\partial \hat{v}_{Tn}}{\partial \hat{z}} \right) \\

&\hat{\dot{w}}_\perp = -\left( \frac{w_\perp}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{t}} + \left( w_\parallel + \frac{\hat{u}_{n\parallel}}{\hat{v}_{Tn}} \right) w_\perp \frac{\partial \hat{v}_{Tn}}{\partial \hat{z}} \right) \\

&\frac{\hat{\dot{F}}_n}{F_n} = \frac{3}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{t}} + \frac{3 (\hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel})}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{z}}
                   - \frac{1}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{t}} - \frac{(\hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel})}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} \\

\hat{\mathcal{C}}_n &= \hat{R}_\mathrm{CX} \hat{n}_i \left( \frac{\hat{v}_{Tn}^3}{\hat{v}_{Ti}^3} F_i - F_n \right) \\
\end{align}
```

### Dimensionless 2D1V ion equations

The moment equations for 2D1V are identical to those for 2D2V, although in 2D1V
$p_{i\perp} = 0$.

#### Full kinetic equation

```math
\begin{align}
\frac{\partial \hat{\bar{f}}_i}{\partial \hat{t}} + \hat{v}_E^r \frac{\partial \hat{\bar{f}}_i}{\partial \hat{r}} + \left(\hat{v}_E^z + b^z \hat{v}_\parallel\right) \frac{\partial \hat{\bar{f}}_i}{\partial \hat{z}}
    - b^z \frac{1}{\hat{m}_i} \frac{\partial\hat{\phi}}{\partial \hat{z}} \frac{\partial \hat{\bar{f}}_i}{\partial \hat{v}_\parallel}
    = \hat{\bar{C}}_{ii}[\hat{\bar{f}}_i, \hat{\bar{f}}_i] - \hat{R}_\mathrm{CX} (\hat{n}_n \hat{\bar{f}}_i - \hat{n}_i \hat{\bar{f}}_n) + \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{\bar{f}}_n + \hat{\bar{S}}_i
\end{align}
```

#### Separate $\hat n_i$

In this subsection $\hat{\bar{F}}_s = c_\mathrm{ref} \bar f_s$.

```math
\begin{align}
& \frac{\partial \hat{\bar{F}}_i}{\partial \hat{t}}
  + \hat{v}_E^r \frac{\partial \hat{\bar{F}}_i}{\partial \hat{r}}
  + (\hat{v}_E^z + b^z \hat{v}_\parallel) \frac{\partial \hat{\bar{F}}_i}{\partial \hat{z}}
  - b^z \frac{1}{\hat{m}_i} \frac{\partial\hat{\phi}}{\partial \hat{z}} \frac{\partial \hat{\bar{F}}_i}{\partial \hat{v}_\parallel} \nonumber \\
& + \left( \frac{(b^z \hat{v}_\parallel - b^z \hat{u}_{i\parallel})}{\hat{n}_i} \frac{\partial \hat{n}_i}{\partial \hat{z}} - b^z \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} + \frac{1}{\hat{n}_i} \hat{S}_{i,n} \right) \hat{\bar{F}}_i \nonumber \\
&\quad= \frac{1}{\hat{n}_i} \int \hat{C}_{ii}[\hat{n}_i \hat{F}_i, \hat{n}_i \hat{F}_i] d^2 \hat{v}_\perp - \hat{R}_\mathrm{CX} \hat{n}_n (\hat{\bar{F}}_i - \hat{\bar{F}}_n) + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{\bar{F}}_n + \frac{1}{\hat{n}_i} \hat{\bar{S}}_i \\
\end{align}
```

#### Separate $\hat n_i$ and $\hat u_{i\parallel}$

In this subsection $\hat{\bar{F}}_s = c_\mathrm{ref} \bar f_s$ and
$\hat{\hat{w}}_\parallel = \hat{w}_\parallel / c_\mathrm{ref}$.

```math
\begin{align}
& \frac{\partial \hat{\bar{F}}_i}{\partial \hat{t}}
  + \hat{v}_E^r \frac{\partial \hat{\bar{F}}_i}{\partial \hat{r}}
  + (\hat{v}_E^z + b^z \hat{\hat{w}}_\parallel + b^z \hat{u}_{i\parallel}) \frac{\partial \hat{\bar{F}}_i}{\partial \hat{z}} \nonumber \\
  &\quad- \left( b^z \hat{w}_\parallel \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}}
                 - b^z \frac{1}{\hat{m}_i \hat{n}_i} \frac{\partial \hat{p}_{i\parallel}}{\partial \hat{z}}
                 + \hat{R}_\mathrm{CX} \hat{n}_n (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
                 + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{u}_{n\parallel}
                 + \frac{1}{\hat{m}_i \hat{n}_i} \hat{S}_{i,\mathrm{mom}}
               \right) \frac{\partial \hat{\bar{F}}_i}{\partial \hat{\hat{w}}_\parallel} \nonumber \\
  &\quad+ \left( b^z \frac{\hat{\hat{w}}_\parallel}{\hat{n}_i} \frac{\partial \hat{n}_i}{\partial \hat{z}}
                 - b^z \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}}
                 + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i}
                 + \frac{1}{\hat{n}_i} \hat{S}_{i,n} \right) \hat{\bar{F}}_i \nonumber \\
&\quad= \frac{1}{\hat{n}_i} \int \hat{C}_{ii}[\hat{n}_i \hat{F}_i, \hat{n}_i \hat{F}_i] d^2 \hat{v}_\perp - \hat{R}_\mathrm{CX} \hat{n}_n (\hat{\bar{F}}_i - \hat{\bar{F}}_n) + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \hat{\bar{F}}_n + \frac{1}{\hat{n}_i} \hat{\bar{S}}_i \\
\end{align}
```

#### Full moment-kinetics (separate $\hat n_i$, $\hat u_{i\parallel}$ and $\hat p_i$)

```math
\begin{align}
&\frac{\partial \bar{F}_i}{\partial \hat{t}} + \hat{\dot{z}} \frac{\partial \bar{F}_i}{\partial \hat{r}} + \hat{\dot{z}} \frac{\partial \bar{F}_i}{\partial \hat{z}} + \hat{\dot{w}}_\parallel \frac{\partial \bar{F}_i}{\partial w_\parallel} + \hat{\dot{w}}_\perp \frac{\partial \bar{F}_i}{\partial w_\perp}
    = \hat{\dot{\bar{F}}}_i + \hat{\bar{\mathcal{C}}}_i + \frac{\hat{v}_{Ti}}{\hat{n}_i} \hat{\bar{S}}_i \\

&\hat{\dot{r}} = \hat{v}_E^r \\

&\hat{\dot{z}} = \hat{v}_E^z + b^z \hat{v}_{Ti} w_\parallel + b^z \hat{u}_{i\parallel} \\

&\hat{\dot{w}}_\parallel =
  \frac{\hat{\dot{v}}_{i\parallel}}{\hat{v}_{Ti}}
  - \frac{1}{\hat{v}_{Ti}} \left( \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{t}}
                                  + \hat{\dot{r}} \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{r}}
                                  + \hat{\dot{z}} \frac{\partial \hat{u}_{i\parallel}}{\partial \hat{z}} \right)
  - \frac{w_\parallel}{\hat{v}_{Ti}} \left( \frac{\partial \hat{v}_{Ti}}{\partial \hat{t}}
                                            + \hat{\dot{r}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{r}}
                                            + \hat{\dot{z}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{z}} \right) \\

&\hat{\dot{v}}_{i\parallel} = - b^z \frac{1}{\hat{m}_i} \frac{\partial \hat{\phi}}{\partial \hat{z}} \\

&\frac{\hat{\dot{\bar{F}}}_i}{\bar{F}_i} =
  \frac{1}{\hat{v}_{Ti}} \left(\frac{\partial \hat{v}_{Ti}}{\partial \hat{t}}
                               + \hat{\dot{r}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{r}}
                               + \hat{\dot{z}} \frac{\partial \hat{v}_{Ti}}{\partial \hat{z}} \right)
  - \frac{1}{\hat{n}_i} \left(\frac{\partial \hat{n}_i}{\partial \hat{t}}
                              + \hat{\dot{r}} \frac{\partial \hat{n}_i}{\partial \hat{r}}
                              + \hat{\dot{z}} \frac{\partial \hat{n}_i}{\partial \hat{z}} \right) \\

&\hat{\bar{\mathcal{C}}}_i =
  \frac{\hat{v}_{Ti}}{\hat{n}_i} \int \hat{C}_{ii}\left[\frac{\hat{n}_i}{\hat{v}_{Ti}^3} F_i, \frac{\hat{n}_i}{\hat{v}_{Ti}^3} F_i\right] d^2 \hat{v}_\perp
  - \hat{R}_\mathrm{CX} \hat{n}_n \left( \bar{F}_i - \frac{\hat{v}_{Ti}}{\hat{v}_{Tn}} \bar{F}_n \right)
  + \hat{R}_\mathrm{ioniz} \frac{\hat{n}_e \hat{n}_n}{\hat{n}_i} \frac{\hat{v}_{Ti}}{\hat{v}_{Tn}} \bar{F}_n \\
\end{align}
```

### Dimensionless 1D1V neutral equations

```math
\begin{align}
&\frac{\partial \hat{n}_n}{\partial \hat{t}} + \frac{\partial}{\partial \hat{z}}\left( \hat{n}_n \hat{u}_{n\parallel} \right)
    = - \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n + \hat{S}_{n,n} \\
& \hat{m}_n \frac{\partial}{\partial \hat{t}}(\hat{n}_n \hat{u}_{n\parallel})
  + \hat{m}_n \frac{\partial}{\partial \hat{z}}(\hat{n}_n \hat{u}_{n\parallel}^2)
  + \frac{\partial \hat{p}_{n\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= -\hat{R}_\mathrm{CX} \hat{m}_n \hat{n}_i \hat{n}_n (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
    - \hat{R}_\mathrm{ioniz} \hat{m}_n \hat{n}_e \hat{n}_n \hat{u}_{n\parallel}
    + \hat{S}_{n,\mathrm{mom}} \\

& \frac{3}{2} \frac{\partial \hat{p}_n}{\partial \hat{t}}
  + \frac{\partial \hat{q}_{n\parallel}}{\partial \hat{z}} + \hat{p}_{n\parallel} \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}}
  + \frac{3}{2} \hat{u}_{n\parallel} \frac{\partial \hat{p}_n}{\partial \hat{z}} + \frac{3}{2} \hat{p}_n \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} \nonumber \\
&\quad= \frac{1}{2} \hat{R}_\mathrm{CX} \hat{n}_i \hat{n}_n \left(3 \hat{T}_i - 3 \hat{T}_n + \hat{m}_i (\hat{u}_{i\parallel} - \hat{u}_{n\parallel})^2 \right)
      - \frac{3}{2} \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{n}_n \hat{T}_n \nonumber \\
&\qquad+ \frac{3}{2} \hat{S}_{n,p} \\
\end{align}
```

#### Full kinetic equation

```math
\begin{align}
& \frac{\partial \hat{\bar{f}}_n}{\partial \hat{t}} + \hat{v}_\parallel \frac{\partial \hat{\bar{f}}_n}{\partial \hat{z}}
    = \hat{R}_\mathrm{CX} (\hat{n}_n \hat{\bar{f}}_i - \hat{n}_i \hat{\bar{f}}_n) - \hat{R}_\mathrm{ioniz} \hat{n}_e \hat{\bar{f}}_n + \hat{\bar{S}}_n \\
\end{align}
```

#### Separate $\hat n_n$

In this subsection $\hat{\bar{F}}_s = c_\mathrm{ref} \bar f_s$.

```math
\begin{align}
& \frac{\partial \hat{\bar{F}}_n}{\partial \hat{t}} + \hat{v}_\parallel \frac{\partial \hat{\bar{F}}_n}{\partial \hat{z}}
  + \left( \frac{(\hat{v}_\parallel - \hat{u}_{n\parallel})}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} - \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} + \frac{1}{\hat{n}_n} \hat{S}_{n,n} \right) \hat{\bar{F}}_n \nonumber \\
&\quad= \hat{R}_\mathrm{CX} \hat{n}_i (\hat{\bar{F}}_i - \hat{\bar{F}}_n) + \frac{1}{\hat{n}_n} \hat{\bar{S}}_n \\
\end{align}
```

#### Separate $\hat n_n$ and $\hat u_{n\parallel}$

In this subsection $\hat{\bar{F}}_s = c_\mathrm{ref} \bar f_s$ and
$\hat{\hat{w}}_\parallel = \hat{w}_\parallel / c_\mathrm{ref}$.

```math
\begin{align}
& \frac{\partial \hat{\bar{F}}_n}{\partial \hat{t}}
  + (\hat{\hat{w}}_\parallel + \hat{u}_{n\parallel}) \frac{\partial \hat{\bar{F}}_n}{\partial \hat{z}} \nonumber \\
  &\quad- \left( \hat{\hat{w}}_\parallel \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}}
                 - \frac{1}{\hat{m}_n \hat{n}_n} \frac{\partial \hat{p}_{n\parallel}}{\partial \hat{z}}
                 - \hat{R}_\mathrm{CX} \hat{n}_i (\hat{u}_{n\parallel} - \hat{u}_{i\parallel})
                 + \frac{1}{\hat{m}_n \hat{n}_n} \hat{S}_{n,\mathrm{mom}}
               \right) \frac{\partial \hat{\bar{F}}_n}{\partial \hat{\hat{w}}_\parallel} \nonumber \\
  &\quad+ \left( \frac{\hat{\hat{w}}_\parallel}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} - \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} + \frac{1}{\hat{n}_n} \hat{S}_{n,n} \right) \hat{\bar{F}}_n \nonumber \\
&\quad= \hat{R}_\mathrm{CX} \hat{n}_i (\hat{\bar{F}}_i - \hat{\bar{F}}_n) + \frac{1}{\hat{n}_n} \hat{\bar{S}}_n \\
\end{align}
```

#### Full moment-kinetics (separate $\hat n_n$, $\hat u_{n\parallel}$ and $\hat p_{n}$)

Note that we do not put a $\hat{\cdot}$ on the already dimensionless quantities
$\bar F_s$, $w_\parallel$, $w_\perp$.

```math
\begin{align}
&\frac{\partial \bar{F}_n}{\partial \hat{t}} + \hat{\dot{z}} \frac{\partial \bar{F}_n}{\partial \hat{z}} + \hat{\dot{w}}_\parallel \frac{\partial \bar{F}_n}{\partial w_\parallel}
    = \hat{\dot{\bar{F}}}_n + \hat{\bar{\mathcal{C}}}_n + \frac{\hat{v}_{Tn}}{\hat{n}_n} \hat{\bar{S}}_n \\

&\hat{\dot{z}} = \hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel} \\

&\hat{\dot{w}}_\parallel = - \left( \frac{1}{\hat{v}_{Tn}} \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{t}} + \left( w_\parallel + \frac{\hat{u}_{n_\parallel}}{\hat{v}_{Tn}} \right) \frac{\partial \hat{u}_{n\parallel}}{\partial \hat{z}} \right. \nonumber \\
    &\qquad\quad  \left. + \frac{w_\parallel}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{t}} + w_\parallel \left( w_\parallel + \frac{\hat{u}_{n\parallel}}{\hat{v}_{Tn}} \right) \frac{\partial \hat{v}_{Tn}}{\partial \hat{z}} \right) \\

&\frac{\hat{\dot{\bar{F}}}_n}{\bar{F}_n} = \frac{1}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{t}} + \frac{(\hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel})}{\hat{v}_{Tn}} \frac{\partial \hat{v}_{Tn}}{\partial \hat{z}}
                   - \frac{1}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{t}} - \frac{(\hat{v}_{Tn} w_\parallel + \hat{u}_{n\parallel})}{\hat{n}_n} \frac{\partial \hat{n}_n}{\partial \hat{z}} \\

&\hat{\bar{\mathcal{C}}}_n = \hat{R}_\mathrm{CX} \hat{n}_i \left( \frac{\hat{v}_{Tn}}{\hat{v}_{Ti}} \bar{F}_i - \bar{F}_n \right) - R_\mathrm{ioniz} n_e \bar{F}_n \\
\end{align}
```

### Conversion to old dimensionless equations

In the 1D1V Excalibur reports, and the original version of the code, a
different set of dimensionless variables were chosen, where
$\check c_\mathrm{ref} = \sqrt{2 \check T_\mathrm{ref} / m_i}$ was used as the
primary reference variable, along with a reference density
$\check n_\mathrm{ref}$, length $\check L_\mathrm{ref}$, and magnetic field
$\check B_\mathrm{ref}$. In the Excalibur reports where a Boltzmann electron
response, which implies constant electron temperature, was used the reference
temperature was taken to be $\check T_\mathrm{ref} = T_e$, but this was later
generalised in the code to an arbitrary $\check T_\mathrm{ref}$ to allow
non-constant electron temperature (with Braginskii or kinetic electrons) or for
varying the constant electron temperature of the Boltzmann response without
re-scaling the rest of the dimensionless variables. Temperatures were
de-dimensionalised using $\check T_\mathrm{ref}$, not $\check m_\mathrm{ref}
\check c_\mathrm{ref}^2$.

In this section, denote dimensionless variables of the original version's
conventions with a $\mathring{\cdot}$.

The 2V distribution function $f_s$ is de-dimensionalised
so that a Maxwellian $\check f_{Ms} = \frac{n_s}{(\pi)^{3/2} \check v_{Ts}}
\exp\left(-v_\parallel^2 / \check v_{Ts}^2 \right)$ would have a maximum value
of $\mathring n_s / \mathring v_{Ts}^3$.

The 1V, marginalised distribution function $\bar f_s$ is de-dimensionalised
so that a 1V Maxwellian $\check{\bar{f}}_{Ms} = \frac{n_s}{\sqrt{\pi} \check
v_{Ts}} \exp\left(-v_\parallel^2 / \check v_{Ts}^2 \right)$ would have a
maximum value of $\mathring n_s / \mathring v_{Ts}$, and its shape function
$\check F_s$ would have a maximum value of 1.

Important to note, in the original version, the dimensionless electrostatic
potential was defined using $\check T_\mathrm{ref}$, as
$\mathring \phi = e \phi / \check T_\mathrm{ref}$, not using
$m_i \check c_\mathrm{ref}^2$ that was used for temperatures.

In the original, for 1D1V runs the input temperatures were $T_{s\parallel}$ not
$T_s$, whereas in the updated version the input temperatures are always $T_s$,
which for 1D1V is $T_s = T_{s\parallel}/3$. However the electron temperature
$T_e$, which is used when calculating the electrostatic potential using the
Boltzmann response, is constant and is effectively always $T_{e\parallel}$
which is assumed equal to $T_e$ for Boltzmann response (i.e. do not assume
$T_{e\perp}=0$ for Boltzmann electrons in 1D1V).

Using these definitions, the old dimensionless variables in terms of the
physical variables and of the current dimensionless variables (where we assume
for the conversion that $\check n_\mathrm{ref} = n_\mathrm{ref}$,
$\check T_\mathrm{ref} = T_\mathrm{ref}$, and
$\check B_\mathrm{ref} = B_\mathrm{ref}$) are, listing the ones with
differences first
```math
\begin{alignat}{2}
\mathring{v}_{Ts} &= \frac{\check{v}_{Ts}}{\check{c}_\mathrm{ref}} = \frac{\sqrt{3} v_{Ts}}{\sqrt{2} c_\mathrm{ref}} = \sqrt{\frac{3}{2}} \hat{v}_{Ts}
    && \hat{v}_{Ts} = \sqrt{\frac{2}{3}} \mathring{v}_{Ts} \\
\mathring{f}_s &= \frac{\pi^{3/2} \check{c}_\mathrm{ref}^3}{\check{n}_\mathrm{ref}} f_s = \frac{(2 \pi)^{3/2} c_\mathrm{ref}^3}{n_\mathrm{ref}} f_s = (2 \pi)^{3/2} \hat{f}_s
    && \hat{f}_s = \frac{1}{(2 \pi)^{3/2}} \mathring{f}_s \\
\mathring{\bar{f}}_s &= \frac{\sqrt{\pi} \check{c}_\mathrm{ref}}{\check{n}_\mathrm{ref}} \bar{f}_s = \frac{\sqrt{2 \pi} c_\mathrm{ref}}{n_\mathrm{ref}} \bar{f}_s = \sqrt{2 \pi} \hat{\bar{f}}_s
    && \hat{\bar{f}}_s = \frac{1}{\sqrt{2 \pi}} \mathring{\bar{f}}_s \\
\mathring{F}_s &= \sqrt{\pi} \check{F}_s = \sqrt{\pi} \sqrt{3} \bar{F}_s
    && \bar{F}_s = \frac{1}{\sqrt{\pi} \sqrt{3}} \mathring{\bar{F}}_s \\
\mathring{f}_\mathrm{Kw} &= \frac{\pi^{3/2} \check{c}_\mathrm{ref}^4}{\check{n}_\mathrm{ref}} f_\mathrm{Kw} = 4\pi^{3/2} \frac{c_\mathrm{ref}^4}{n_\mathrm{ref}} f_\mathrm{Kw} = 4\pi^{3/2} \hat{f}_\mathrm{Kw}
    && \hat{f}_\mathrm{Kw} = \frac{1}{4\pi^{3/2}} \mathring{f}_\mathrm{Kw} \\
\mathring{\bar{f}}_\mathrm{Kw} &= \sqrt{\pi} \check{c}_\mathrm{ref}^2 \bar{f}_\mathrm{Kw} = 2\sqrt{\pi} c_\mathrm{ref} \bar{f}_\mathrm{Kw} = 2\sqrt{\pi} \hat{\bar{f}}_\mathrm{Kw}
    && \hat{\bar{f}}_\mathrm{Kw} = \frac{1}{2\sqrt{\pi}} \mathring{\bar{f}}_\mathrm{Kw} \\
\mathring{t} &= \frac{\check{c}_\mathrm{ref} t}{\check{L}_\mathrm{ref}} = \sqrt{2} \frac{c_\mathrm{ref} t}{L_\mathrm{ref}} = \sqrt{2} \hat{t}
    && \hat{t} = \frac{1}{\sqrt{2}} \mathring{t} \\
\frac{\partial}{\partial \mathring{t}} &= \frac{\check{L}_\mathrm{ref}}{\check{c}_\mathrm{ref}} \frac{\partial}{\partial t} = \frac{L_\mathrm{ref}}{\sqrt{2} c_\mathrm{ref}} \frac{\partial}{\partial \hat{t}} = \frac{1}{\sqrt{2}} \frac{\partial}{\partial \hat{t}}
    && \frac{\partial}{\partial \hat{t}} = \sqrt{2} \frac{\partial}{\partial \mathring{t}} \\
\mathring{v}_\parallel &= \frac{v_\parallel}{\check{c}_\mathrm{ref}} = \frac{v_\parallel}{\sqrt{2} c_\mathrm{ref}} = \frac{\hat{v}_\parallel}{\sqrt{2}}
    && \hat{v}_\parallel = \sqrt{2} \mathring{v}_\parallel \\
\frac{\partial}{\partial \mathring{v}_\parallel} &= \check{c}_\mathrm{ref} \frac{\partial}{\partial v_\parallel} = \sqrt{2} c_\mathrm{ref} \frac{\partial}{\partial v_\parallel} = \sqrt{2} \frac{\partial}{\partial \hat{v}_\parallel}
    && \frac{\partial}{\partial \hat{v}_\parallel} = \frac{1}{\sqrt{2}} \frac{\partial}{\partial \mathring{v}_\parallel} \\
\mathring{v}_\perp &= \frac{v_\perp}{\check{c}_\mathrm{ref}} = \frac{v_\perp}{\sqrt{2} c_\mathrm{ref}} = \frac{\hat{v}_\perp}{\sqrt{2}}
    && \hat{v}_\perp = \sqrt{2} \mathring{v}_\perp \\
\frac{\partial}{\partial \mathring{v}_\perp} &= \check{c}_\mathrm{ref} \frac{\partial}{\partial v_\perp} = \sqrt{2} c_\mathrm{ref} \frac{\partial}{\partial v_\perp} = \sqrt{2} \frac{\partial}{\partial \hat{v}_\perp}
    && \frac{\partial}{\partial \hat{v}_\perp} = \frac{1}{\sqrt{2}} \frac{\partial}{\partial \mathring{v}_\perp} \\
\mathring{w}_\parallel &= \frac{\mathring{v}_\parallel}{\mathring{v}_{Ts}} = \frac{v_\parallel}{\check{v}_{Ts}} = \frac{v_\parallel}{\sqrt{3}v_{Ts}} = \frac{w_\parallel}{\sqrt{3}}
    && w_\parallel = \sqrt{3} \mathring{w}_\parallel \\
\mathring{u}_{s\parallel} &= \frac{u_{s\parallel}}{\check{c}_\mathrm{ref}} = \frac{u_{s\parallel}}{\sqrt{2} c_\mathrm{ref}} = \frac{\hat{u}_{s\parallel}}{\sqrt{2}}
    && \hat{u}_{s\parallel} = \sqrt{2} \mathring{u}_{s\parallel} \\
\mathring{p}_s &= \frac{p_s}{\check{n}_\mathrm{ref} m_i \check{c}_\mathrm{ref}^2} = \frac{p_s}{\check{n}_\mathrm{ref} m_i 2 \check{T}_\mathrm{ref}} = \frac{\hat{p}_s}{2}
    && \hat{p}_s = 2 \mathring{p}_s \\
\mathring{q}_{s\parallel} &= \frac{\check{q}_{s\parallel}}{m_i \check{n}_\mathrm{ref} \check{c}_\mathrm{ref}^3} = \frac{2 q_{s\parallel}}{m_i n_\mathrm{ref} 2^{3/2} c_\mathrm{ref}^3} = \frac{\hat{q}_{s\parallel}}{\sqrt{2}}
    && \hat{q}_{s\parallel} = \sqrt{2}\mathring{q}_{s\parallel} \\
\mathring{R}_\mathrm{CX} &= \frac{\check{L}_\mathrm{ref} \check{n}_\mathrm{ref} R_\mathrm{CX}}{\check{c}_\mathrm{ref}} = \frac{L_\mathrm{ref} n_\mathrm{ref} R_\mathrm{CX}}{\sqrt{2} c_\mathrm{ref}} = \frac{1}{\sqrt{2}} \hat{R}_\mathrm{CX}
    && \hat{R}_\mathrm{CX} = \sqrt{2} \mathring{R}_\mathrm{CX} \\
\mathring{R}_\mathrm{ioniz} &= \frac{\check{L}_\mathrm{ref} \check{n}_\mathrm{ref} R_\mathrm{ioniz}}{\check{c}_\mathrm{ref}} = \frac{L_\mathrm{ref} n_\mathrm{ref} R_\mathrm{ioniz}}{\sqrt{2} c_\mathrm{ref}} = \frac{1}{\sqrt{2}} \hat{R}_\mathrm{ioniz} \quad
    && \hat{R}_\mathrm{ioniz} = \sqrt{2} \mathring{R}_\mathrm{ioniz} \\
\mathring{S}_s &= \frac{\pi^{3/2} \check{L}_\mathrm{ref} \check{c}_\mathrm{ref}^2 S_s}{\check{n}_\mathrm{ref}} = \frac{\pi^{3/2} L_\mathrm{ref} 2 c_\mathrm{ref}^2 S_s}{n_\mathrm{ref}} = 2 \pi^{3/2} \hat{S}_s
    && \hat{S}_s = \frac{1}{2 \pi^{3/2}} \mathring{S}_s \\
\mathring{\bar{S}}_s &= \frac{\sqrt{\pi} \check{L}_\mathrm{ref} \bar{S}_s}{\check{n}_\mathrm{ref}} = \sqrt{\pi} \hat{\bar{S}}_s
    && \hat{\bar{S}}_s = \frac{1}{\sqrt{\pi}} \mathring{\bar{S}}_s \\
\mathring{S}_{s,n} &= \frac{\check{L}_\mathrm{ref} S_{s,n}}{\check{c}_\mathrm{ref} \check{n}_\mathrm{ref}} = \frac{L_\mathrm{ref} S_{s,n}}{\sqrt{2} c_\mathrm{ref} n_\mathrm{ref}} = \frac{1}{\sqrt{2}} \hat{S}_{s,n}
    && \hat{S}_{s,n} = \sqrt{2} \mathring{S}_{s,n} \\
\mathring{S}_{s,\mathrm{mom}} &= \frac{\check{L}_\mathrm{ref} S_{s,\mathrm{mom}}}{\check{m}_\mathrm{ref} \check{n}_\mathrm{ref} \check{c}_\mathrm{ref}^2} = \frac{L_\mathrm{ref} S_{s,\mathrm{mom}}}{2 m_\mathrm{ref} n_\mathrm{ref} c_\mathrm{ref}^2} = \frac{1}{2} \hat{S}_{s,\mathrm{mom}}
    && \hat{S}_{s,\mathrm{mom}} = 2 \mathring{S}_{s,\mathrm{mom}} \\
\mathring{S}_{s,E} &= \frac{\check{L}_\mathrm{ref} S_{s,E}}{\check{m}_\mathrm{ref} \check{n}_\mathrm{ref} \check{c}_\mathrm{ref}^3} = \frac{L_\mathrm{ref} S_{s,E}}{2^{3/2} m_\mathrm{ref} n_\mathrm{ref} c_\mathrm{ref}^2} = \frac{1}{2 \sqrt{2}} \hat{S}_{s,E}
    && \hat{S}_{s,E} = 2 \sqrt{2} \mathring{S}_{s,E} \\
\mathring{S}_{s,p} &= \frac{\check{L}_\mathrm{ref} S_{s,p}}{\check{m}_\mathrm{ref} \check{n}_\mathrm{ref} \check{c}_\mathrm{ref}^3} = \frac{L_\mathrm{ref} S_{s,p}}{2^{3/2} m_\mathrm{ref} n_\mathrm{ref} c_\mathrm{ref}^2} = \frac{1}{2 \sqrt{2}} \hat{S}_{s,p}
    && \hat{S}_{s,p} = 2 \sqrt{2} \mathring{S}_{s,p} \\
\mathring{S}_{s,p_\parallel} &= \frac{1}{2 \sqrt{2}} \hat{S}_{s,p_\parallel} = \frac{3}{2 \sqrt{2}} \hat{S}_{s,p}
    && \hat{S}_{s,p} = \frac{2 \sqrt{2}}{3} \mathring{S}_{s,p_\parallel} \\
\mathring{\gamma}_{ss'} &= \frac{\check{n}_\mathrm{ref} \check{L}_\mathrm{ref}}{\check{m}_\mathrm{ref}^2 \check{c}_\mathrm{ref}^4} \gamma_{ss'} = \frac{n_\mathrm{ref} L_\mathrm{ref}}{4 m_\mathrm{ref}^2 c_\mathrm{ref}^4} \gamma_{ss'} = \frac{1}{4} \hat{\gamma}_{ss'}
    && \hat{\gamma}_{ss'} = 4 \mathring{\gamma}_{ss'} \\
\mathring{n}_s &= \frac{n_s}{\check{n}_\mathrm{ref}} = \hat{n}_s
    && \hat{n}_s = \mathring{n}_s \\
\mathring{T}_s &= \frac{T_s}{\check{T}_\mathrm{ref}} = \frac{T_s}{\check{T}_\mathrm{ref}} = \hat{T}_s
    && \hat{T}_s = \mathring{T}_s \\
\mathring{L}_z &= \frac{L_z}{\check{L}_\mathrm{ref}} = \hat{L}_z
    && \hat{L}_z = \mathring{L}_z \\
\mathring{B} &= \frac{B}{\check{B}_\mathrm{ref}} = \hat{B}
    && \hat{B} = \mathring{B} \\
\mathring{m}_i &= \frac{m_i}{m_i} = 1 = \hat{m}_i
    && \hat{m}_i = \mathring{m}_i \\
\mathring{m}_e &= \frac{m_e}{m_i} = \hat{m}_e
    && \hat{m}_e = \mathring{m_e} \\
\mathring{\phi} &= \frac{e \phi}{\check{T}_\mathrm{ref}} = \hat{\phi}
    && \hat{\phi} = \mathring{\phi} \\
\mathring{E}_\parallel &= \frac{e \check{L}_\mathrm{ref} E_\parallel}{\check{T}_\mathrm{ref}} = \hat{E}_\parallel
    && \hat{E}_\parallel = \mathring{E}_\parallel \\
\mathring{z} &= \frac{z}{\check{L}_\mathrm{ref}} = \hat{z}
    && \hat{z} = \mathring{z} \\
\frac{\partial}{\partial \mathring{z}} &= \check{L}_\mathrm{ref} \frac{\partial}{\partial z} = \frac{\partial}{\partial \hat{z}}
    && \frac{\partial}{\partial \hat{z}} = \frac{\partial}{\partial \mathring{z}} \\
\end{alignat}
```
In the 'old' equations a parallel-pressure source term
$S_{s,p_\parallel} = \int m_s (v_\parallel - u_{s\parallel})^2 \bar S_s dv_\parallel$
was used in the parallel pressure equation. In the 1D1V limit the pressure source is
$S_{s,p} = \int \frac{1}{3} m_s (v_\parallel - u_{s\parallel})^2 \bar S_s dv_\parallel = S_{s,p_\parallel}/3$.

Old 1D1V moment kinetic equations
---------------------------------

These were the definitions and dimensionless variables used before PR #322, April 2025.

```@raw html
<details>
<summary style="text-align:center">[ notes using old definitions and dimensionless variables ]</summary>
```
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

### Moment equations

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

### Kinetic equation

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
```@raw html
</details>
```
