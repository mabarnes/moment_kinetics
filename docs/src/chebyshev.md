Chebyshev tranform via Fourier transform
===============================================

```math
\begin{equation}
\end{equation}
```
We express a function $f$ as a sum of Chebyshev polynomials
```math 
\begin{equation} f(x) = \sum^N_{n=0} a_{n}T_n(x)\label{eq:cheb-expansion}\end{equation}
```
The Chebyshev polynomials are defined by 
```math
\begin{equation} T_n(\cos \theta) = \cos n \theta, {\rm~with~}x = \cos \theta. \end{equation}
```
We can see how to find $\{a_{n}\}$ given $\{f(x_j)\}$ via Fourier transform. 
The Fourier series representation of $f$ on a uniform grid indexed by $j$ is defined by 
```math
\begin{equation} f_j = \sum_{k=0}^{M-1} b_{k}\exp\left[i \frac{2\pi k j}{M}\right].\label{eq:fourier-series}\end{equation}
```

Gauss-Chebyshev-Lobotto points
===============================================

We pick points 
```math
\begin{equation} x_j = \cos \theta_j, \quad \theta_j = \frac{j \pi}{N} \quad 0 \leq j \leq N.\end{equation}
```
Then 
```math
\begin{equation} T_n(x_j) = \cos \frac{n j \pi}{N}.\end{equation}
```
Assuming that $M = 2N$, with $N$ an integer, and $b_{k} = b_{M-k}$ for $k>0$, we have that 
```math
\begin{equation} f_j = b_{0} + b_{N}(-1)^j + \sum_{n=1}^{N-1}
b_{n}\left(\exp\left[i \frac{\pi n j}{N}\right]+\exp\left[-i \frac{\pi n j}{N}\right]\right).\end{equation}
```
Comparing this to the expression for $f(x_j)$ in the Chebyshev representation, 
using the form of $T_n(x_j)$, 
```math
\begin{equation} f_j = a_{0} + a_{N}(-1)^j + \frac{1}{2}\sum_{n=1}^{N-1}
a_{n}\left(\exp\left[i \frac{\pi n j}{N}\right]+\exp\left[-i \frac{\pi n j}{N}\right]\right),\end{equation}
```
we find that the Chebyshev representation on the Chebyshev points is equivalent 
to the Fourier representation on the uniform grid points, if we identify
```math
\begin{equation} b_{0} = a_{0}, \quad  b_{N} = a_{N}, \quad b_{j} = \frac{a_{j}}{2} {\rm~for~} 1 \leq j \leq N-1. \end{equation}
```
This fact allows us to carry out the Chebyshev tranform by Fourier transforming the $\{f_j\}$ data
and carrying out the correct normalisation of the resulting coefficients. 

Gauss-Chebyshev-Radau points
===============================================

The last subsection dealt with grids which contain both endpoints on the $[-1,1]$ domain. 
Certain problems require domains which contain a single endpoint, i.e., $x \in (-1,1]$. For 
these cases we choose the points 
```math
\begin{equation} x_j = \cos \theta_j, \quad \theta_j = \frac{2 j \pi}{2 N + 1} \quad 0 \leq j \leq N.\end{equation}
```
Writing out the Chebyshev series \eq{eq:cheb-expansion}, 
we have that 
```math
\begin{equation} \begin{split} f(x_j) = & \sum^N_{n=0} a_{n} \cos \frac{2 n j \pi}{2 N + 1} \\ & = a_{0} + \sum^N_{n=1} \frac{a_{n}}{2}\left(\exp\left[i \frac{2\pi n j}{2N +1}\right] + \exp\left[-i \frac{2\pi n j}{2N +1}\right]\right).\end{split} \label{eq:cheb-expansion-radau-points}\end{equation}
```
The form of the series \eq{eq:cheb-expansion-radau-points} is identical to the form of 
a Fourier series on an odd number of points, i.e., taking $M = 2 N + 1$ in equation \eq{eq:fourier-series},
and assuming $b_{k} = b_{M -k}$ for $k>1$,
we have that 
```math
\begin{equation} f_j = b_{0} + \sum_{k=1}^{N} b_{k}\left(\exp\left[i \frac{2\pi k j}{2N+1}\right] + \exp\left[-i \frac{2\pi k j}{2N+1} \right]\right). \end{equation}
```
We can thus take a Chebyshev transform using a Fourier transform on Gauss-Chebyshev-Radau points if we identify 
```math
\begin{equation} b_{0} = a_{0}, \quad b_{j} = \frac{a_{j}}{2} {\rm~for~} 1 \leq j \leq N. \end{equation}
```

Chebyshev coefficients of derivatives of a function
===============================================

Starting from the expression of $f$ as a sum Chebyshev polynomials, equation \eq{eq:cheb-expansion},
we can obtain an expression for the derivative
```math
\begin{equation} \frac{d f}{d x} = \sum^N_{n=0} a_{n}\frac{d T_{n}}{d x}. \label{eq:derivative-def}\end{equation}
```
We note that we must be able to express ${d f}/{d x}$ as a sum 
of Chebyshev polynomials of up to order $N-1$, i.e.,
```math
\begin{equation} \frac{d f}{d x} = \sum^{N-1}_{n=0} d_{n}T_{n}. \end{equation}
```
We must determine the set $\{d_{n}\}$ in terms of the set $\{a_{n}\}$.
First, we equate the two expressions to find that 
```math
\begin{equation} \sum^N_{k=0} a_{k}\frac{d T_{k}}{d x} = \sum^{N-1}_{n=0} d_{n}T_{n}. \label{eq:dn-def}\end{equation}
```
We use the Chebyshev polynomials of the second kind $U_n{x}$ to aid us in the calculation of the set $\{d_{n}\}$. 
These polynomials are defined by 
```math
\begin{equation} U_{0}(x) = 1, \quad U_{1}(x) = 2x, \quad U_{n+1} = 2 x U_{n}(x) - U_{n-1}(x).\end{equation}
```
Note the useful relations 
```math
\begin{equation} \frac{d T_{n}}{d x} = n U_{n-1}, {\rm~for~}n\geq 1, \quad \frac{d T_{0}}{d x} = 0,\end{equation}
```
```math
\begin{equation} T_{n} = \frac{1}{2}\left(U_{n} - U_{n -2}\right), T_{0} = U_{0}\quad, {\rm ~and~} \quad 2 T_{1} = U_{1}. \end{equation}
```
Using these identities, which may be obtained from the trigonometric definition of $U_{n}(\cos \theta)$
```math
\begin{equation}  U_{n}(\cos \theta) \sin \theta = \sin \left((n+1)\theta\right),\end{equation}
```
we find that equation  \eq{eq:dn-def} becomes 
```math
\begin{equation} \begin{split}\sum^N_{n=1} a_{n} n U_{n-1}(x) =& \frac{d_{N-1}}{2}U_{N-1}+\frac{d_{N-2}}{2}U_{N-2} 
\\ & + \sum^{N-3}_{k=1} \frac{d_{k}-d_{k+2}}{2}U_{k} + \left(d_{0} - \frac{d_{2}}{2}\right)U_{0}. \end{split}
\label{eq:dn-def-U}\end{equation}
```
Using the orthogonality relation 
```math
\begin{equation} \int^1_{-1} U_{m}(x)U_{n}(x)\sqrt{1-x^2} \; d x = 
\left\{\begin{tabular}{l} $0 {\rm ~if~} n\neq m $ \\ $\pi/2 {\rm ~if~} n=m$  \end{tabular}\right.,\end{equation}
```
we obtain the (unqiuely-determined) relations 
```math
\begin{equation} \begin{split} &d_{N-1} = 2Na_{N},\quad d_{N-2} = 2(N-1)a_{N-1}, \\ 
& d_{k} = 2(k+1) a_{k+1} + d_{k+2}, \quad d_{0} = \frac{d_{2}}{2} + a_{1}.\end{split} \label{eq:dn-result-U}\end{equation}
```       

Clenshaw-Curtis integration weights
===============================================

We require the integration weights for the set of points $\{x_j\}$ chosen 
in our numerical scheme. The weights $w_{j}$ are defined implicitly by 
```math
\begin{equation} \int^{1}_{-1} f(x) \; d x = \sum_{j=0}^N f(x_j) w_{j}. \label{eq:w-sum}\end{equation}
```
In the Chebyshev scheme we use the change of variables $x = \cos \theta$
to write 
```math
\begin{equation} \int^{1}_{-1} f(x) \; d x = \int^\pi_0 f(\cos\theta) \sin \theta \; d \theta . \label{eq:change-of-variables-integral} \end{equation}
```
 Using the series expansion \eq{eq:cheb-expansion} in equation \eq{eq:change-of-variables-integral}
 we find that 
 ```math
 \begin{equation} \int^{1}_{-1} f(x) \; d x = \sum^N_{n=0} a_{n}\int^\pi_0 \cos (n \theta) \sin \theta \; d \theta
 . \label{eq:series-integral} \end{equation}
 ```
 Note the integral identity
 ```math
\begin{equation} \int^\pi_0 \cos(n \theta) \sin \theta \; d \theta = \frac{\cos(n \pi) +1}{1 - n^2} {\rm~for~} n \geq 0.\end{equation}
```
 Also note that 
 ```math
 \begin{equation} \frac{\cos(n \pi) +1}{1 - n^2} = \left\{\begin{tabular}{l} $0 {\rm ~if~} n = 2 r + 1, ~r \in \mathbb{Z} $ \\ $2/(1 - n^2) {\rm ~if~} n=2r,~r. \in \mathbb{Z}$  \end{tabular}\right. \end{equation}
 ```
 We define ```math
 \begin{equation} J_{n} = \frac{\cos(n \pi) +1}{1 - n^2}. \end{equation}
 ```
 Using this definition, we can write the integral of $f(x)$ can be written 
 in terms of a sum over of the Chebyshev coefficients:
 ```math
 \begin{equation} \int^{1}_{-1} f(x) \; d x = \sum_{n=0}^N J_{n} a_{n}. \label{eq:Cheb-sum}\end{equation}
 ```
 
 To avoid computing the set of coefficients $\{a_{n}\}$ every time we wish to integrate $f(x_j)$,
 we use the inverse transforms. This transform allows us to rewrite equation \eq{eq:Cheb-sum} in the form \eq{eq:w-sum}.
 Since the inverse transform differs between the Gauss-Chebyshev-Lobotto and Gauss-Chebyshev-Radau cases, we treat each 
 case separately below. 
 
Weights on Gauss-Chebyshev-Lobotto points
===============================================
  We use the inverse transformation 
 ```math
 \begin{equation} a_{n} = \frac{q_{n}}{2N}\sum^{2N-1}_{j=0} \hat{f}_j \exp\left[- i \frac{2\pi n j}{2N}\right], \label{eq:inverse-transform-GCL}\end{equation}
 ```
 where 
 ```math
 \begin{equation} q_{n} = \left\{\begin{tabular}{l} $2 {\rm ~if~} n\neq0,N $ \\ $1 {\rm ~if~} n=0,N$  \end{tabular}\right.,\end{equation}
 ```
and $\hat{f}_j$ is $f(x_j)$ on the extended domain in FFT order, i.e.,
```math
\begin{equation} \hat{f}_j = f(x_{j}) {\rm~for~} 0 \leq j \leq N ,quad \hat{f}_j = f(x_{2N-j}){\rm~for~} N+1 \leq j \leq 2N-1. \end{equation}
```
 With this inverse tranformation, we can write 
```math
\begin{equation} \begin{split}\sum_{n=0}^N J_{n} a_{n} & =  \sum^{2N-1}_{n=0} \frac{a_{n}J_{n}}{q_{n}} \\
 & = \sum^{2N-1}_{j=0}\sum^{2N-1}_{n=0} \frac{\hat{f}_j J_{n}}{2N} \exp\left[-i \frac{2\pi n j}{2N}\right] \\ 
 & = \sum^{2N-1}_{j=0} \hat{f}_j v_{j} = \sum^{N}_{j=0} \hat{f}_j q_{j}v_{j},\end{split}\label{eq:weights-working}\end{equation}
```
 where in the first step we have extended the sum from $N$ to $2N-1$ and used FFT-order definitions of $J_{n}$ and $a_{n}$
```math
\begin{equation} J_{j} = J_{2N-j}, {\rm~for~} N+1 \leq j \leq 2N-1,\end{equation}
```
```math
\begin{equation} a_{j} = a_{2N-j}, {\rm~for~} N+1 \leq j \leq 2N-1.\end{equation}
```
In the second step we use the definition of the inverse transform \eq{eq:inverse-transform-GCR}, and 
in the third step we define 
```math
\begin{equation} v_{j} = \sum_{n=0}^{2N-1}\frac{J_{n}}{2N}\exp\left[-i \frac{2\pi n j}{2N}\right].\end{equation}
```
Finally, we can compare equations \eq{eq:w-sum} and \eq{eq:weights-working} and deduce that 
```math
\begin{equation} w_{j} = q_{j}v_{j} {\rm~for~} 0 \leq j \leq N.  \end{equation}
```
We can write $v_{j}$ in terms of a discrete cosine transform, i.e.,
```math
\begin{equation} v_{j} = \frac{1}{2N}\left(J_{0} + (-1)^jJ_{N} + 2\sum_{n=1}^{N-1}J_{n}\cos\left(\frac{\pi n j}{N}\right)\right).\end{equation}
```
 
Weights on Gauss-Chebyshev-Radau points
===============================================
We use the inverse transformation 
```math
\begin{equation} a_{n} = \frac{q_{n}}{2N+1}\sum^{2N}_{j=0} \hat{f}_j \exp\left[- i \frac{2\pi n j}{2N+1}\right], \label{eq:inverse-transform-GCR}\end{equation}
```
where 
```math
\begin{equation} q_{n} = \left\{\begin{tabular}{l} $2 {\rm ~if~} n > 0 $ \\ $1 {\rm ~if~} n=0$  \end{tabular}\right.,\end{equation}
```
and $\hat{f}_j$ is $f(x_j)$ on the extended domain in FFT order, i.e.,
```math
\begin{equation} \hat{f}_j = f(x_{j}) {\rm~for~} 0 \leq j \leq N ,\quad \hat{f}_j = f(x_{2N-j+1}){\rm~for~} N+1 \leq j \leq 2N. \end{equation}
```
Note that the details of what is the appropriate FFT order depends on the order in which the points $x_j$ are stored.
The key detail in the Chebyshev-Radau scheme is that (in the notation above)
$x_0 = 1$ is not a repeated point, and must occupy $\hat{f}_0$. 
With this inverse tranformation, we can write 
```math
\begin{equation} \begin{split}\sum_{n=0}^N J_{n} a_{n} & =  \sum^{2N}_{n=0} \frac{a_{n}J_{n}}{q_{n}} \\
& = \sum^{2N}_{j=0}\sum^{2N}_{n=0} \frac{\hat{f}_j J_{n}}{2N+1} \exp\left[-i \frac{2\pi n j}{2N+1}\right] \\ 
& = \sum^{2N}_{j=0} \hat{f}_j v_{j} = \sum^{N}_{j=0} \hat{f}_j q_{j}v_{j},\end{split}\label{eq:weights-working}\end{equation}
```
where in the first step we have extended the sum from $N$ to $2N$ and used FFT-order definitions of $J_{n}$ and $a_{n}$
```math
\begin{equation} J_{j} = J_{2N+1-j}, {\rm~for~} N+1 \leq j \leq 2N,\end{equation}
```
```math
\begin{equation} a_{j} = a_{2N+1-j}, {\rm~for~} N+1 \leq j \leq 2N.\end{equation}
```
In the second step we use the definition of the inverse transform \eq{eq:inverse-transform-GCR}, and 
in the third step we define 
```math
\begin{equation} v_{j} = \sum_{n=0}^{2N}\frac{J_{n}}{2N+1}\exp\left[-i \frac{2\pi n j}{2N+1}\right].\end{equation}
```
Finally, we can compare equations \eq{eq:w-sum} and \eq{eq:weights-working} and deduce that 
```math
\begin{equation} w_{j} = q_{j}v_{j} {\rm~for~} 0 \leq j \leq N.  \end{equation}
```
