smc-toyexample
==============

Sequential Monte Carlo methods (particle filtering/smoothing) for a toy problem of the following form

- x_{t+1} = a x_t + v_t,  v_t \sim N(0,\sigma^2_v),
- y_t = x_t + e_t,  e_t \sim N(0,\sigma^2_e)

where a denotes the scale parameter and the noise variances are given by \sigma^2_v and \sigma^2_e.

Files
-------------
The following files are included
- toyproblem_pf: estimate the states given a data realisation and the parameters using a bootstrap particle filter.
- toyproblem_fl: estimate the states given a data realisation and the parameters using a fixed-lag smoother.
- toyproblem_ffbsi: estimate the states given a data realisation and the parameters using a forward-filtering backward-simulator.
- toyproblem_score: estimate the score function given a data realisation using a forward-filtering backward-smoother.
