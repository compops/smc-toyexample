smc-toyexample
==============

Sequential Monte Carlo methods (particle filtering/smoothing) for a toy problem of the following form

- x(t+1) = a x(t) + v(t),  v(t) ~ N(0,sigmav^2),
- y(t)   = c x(t) + e(t),  e(t) ~ N(0,sigmae^2)

where a and c denote scale parameters and the noise variances are given by sigmav^2 and sigmae^2.

Files
-------------
The following files are included
- toyex_pf: estimates the states given a data realisation and the parameters using a bootstrap particle filter.
- toyex_bpf: estimates the states given a data realisation and the parameters using a information filter (backward particle filter).
- toyex_fl: estimates the states given a data realisation and the parameters using a fixed-lag smoother.
- toyex_ffbsm: estimates the states given a data realisation and the parameters using a forward-filtering backward-smoothing.
