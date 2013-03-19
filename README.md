smc-toyexample
==============

Sequential Monte Carlo methods (particle filtering/smoothing) for a toy problem of the following form

x_{t+1} = a x_{t+1} + v_t,  v_t \sim N(0,\sigma^2_v)
y_{t} = x_{t+1} + e_t,  e_t \sim N(0,\sigma^2_e)

where a denotes the scale parameter and the noise variances are given by \sigma^2_v and \sigma^2_e.
