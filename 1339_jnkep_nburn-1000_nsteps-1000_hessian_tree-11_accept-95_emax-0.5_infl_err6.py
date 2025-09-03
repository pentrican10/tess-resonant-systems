import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm
import matplotlib

import jax.numpy as jnp
from jax import config, random
import numpyro, jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
config.update('jax_enable_x64', True)
numpyro.set_platform('cpu') 
num_chains = 4
numpyro.set_host_device_count(num_chains)
print ('# jax device count:', jax.local_device_count())

from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv import ttv_default_parameter_bounds, ttv_optim_curve_fit, scale_pdic
import corner



d = pd.read_csv("toi_1339_transit_data_all.txt", sep="\s+", header=0, names=['Planet_num', 'Index', 'Tc', 'Tc_err'])

### get times, errs from the data
list_of_obs_transit_times = []
list_of_transit_time_errs = []

period_guess = [28.579657965796578, 38.35013501350135]
for j in range(2):
    list_of_obs_transit_times.append(np.array(d.Tc[d.Planet_num==j+2]))
    list_of_transit_time_errs.append(np.array(d.Tc_err[d.Planet_num==j+2]))
# index_obs_1 = np.array(d.Index[d.Planet_num==1])
index_obs_2 = np.array(d.Index[d.Planet_num==2])
index_obs_3 = np.array(d.Index[d.Planet_num==3])

tcobs = []
errorobs = []
for j in range(2):
    tcobs.append(np.array(d.Tc[d.Planet_num == j + 2]))
    errorobs.append(np.array(d.Tc_err[d.Planet_num == j + 2]))
# Inflate errors by factor of 3
inflation_factor = 6
inflated_errorobs = [errs * inflation_factor for errs in errorobs]
inflated_list_of_transit_time_errs = [errs * inflation_factor for errs in list_of_transit_time_errs]


### run JaxTTV sim
t_start = 1723.  # start of integration
# t_end = 5500. # end of integration
t_end = 3670.
dt = 0.9 # integration timestep


jttv = JaxTTV(t_start, t_end, dt, tcobs, period_guess, errorobs=inflated_errorobs, print_info=True)



### set bounds for fit
param_bounds = ttv_default_parameter_bounds(jttv, emax=0.5)

### initialize p_init: p1,p2,ecosw1,ecosw2,esinw1,esinw2,tic1,tic2,lnpmass1,lnpmass2
### using best fit for run with all points
p_init = np.array([28.579657965796578, 38.35013501350135, 
                   -0.021369753, -0.018361236, 
                   -0.023016884, -0.019734158, 
                   1726.05430737, 1743.55228223, 
                   -10.97106092, -11.30491702])

### fit
popt = ttv_optim_curve_fit(jttv,param_bounds,p_init=p_init, plot=False)
# plt.show()
print(popt)


tcall = jttv.get_transit_times_all_list(popt,truncate=False)
# jttv.plot_model(tcall, marker='.')
# plt.show()


### check precision and residuals 
tc, _ = jttv.check_timing_precision(popt)
jitter = 5 / (24*60)
jitters = [jitter, jitter]
pdic_normal, pdic_student = jttv.check_residuals(popt)#, jitters=jitters)
# plt.show()

import pickle
print(popt)
print(param_bounds)
with open('1339_jnkep_initial_fit_data_hessian_tree-11_accept-95_emax-0.5_infl_err6.pkl', 'wb') as f:
    pickle.dump({'jttv': jttv, 'popt': popt, 'param_bounds': param_bounds}, f)

print('Initial fit data saved successfully')




def model_scaled(sample_keys, param_bounds):
    """numpyro model for scaled parameters"""
    par = {}

    # sample parameters from priors
    for key in sample_keys:
        par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds[key][0]*0, param_bounds[key][0]*0+1.))
        par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])
    if "pmass" not in sample_keys:
        par["pmass"] = numpyro.deterministic("pmass", jnp.exp(par["lnpmass"]))
    
    # Jacobian for uniform ecc prior
    ecc = numpyro.deterministic("ecc", jnp.sqrt(par['ecosw']**2+par['esinw']**2))
    numpyro.factor("eprior", -jnp.log(ecc))

    # compute transit times
    tcmodel, ediff = jttv.get_transit_times_obs(par)
    numpyro.deterministic("ediff", ediff)
    numpyro.deterministic("tcmodel", tcmodel)
    
    # likelihood
    tcerrmodel = jttv.errorobs_flatten     
    numpyro.sample("obs", dist.Normal(loc=tcmodel, scale=tcerrmodel), obs=jttv.tcobs_flatten)


# physical parameters to sample from
sample_keys = ["ecosw", "esinw", "pmass", "period", "tic"] 

# uniform mass prior# scaled parameters
pdic_scaled = scale_pdic(popt, param_bounds)


### initializing mass matrix
# information matrix and parameter covariance
from jnkepler.jaxttv.information import information
fisher_information = information(jttv, popt, sample_keys)
parameter_cov = jnp.linalg.inv(fisher_information)

for i,key in enumerate(sample_keys):
    print(key, np.diag(jnp.sqrt(parameter_cov))[3*i:3*i+3])

# Fisher information and parameter covariance matrix for scaled parameters
fisher_information_scaled = information(jttv, popt, sample_keys, param_bounds=param_bounds)
parameter_cov_scaled = jnp.linalg.inv(fisher_information_scaled)

# initialize inverse mass matrix using parameter covariance
dense_mass = [tuple([key+"_scaled" for key in sample_keys])]
inverse_mass_matrix = {dense_mass[0]: parameter_cov_scaled}

kernel = NUTS(model_scaled, 
            init_strategy=init_to_value(values=pdic_scaled), 
            dense_mass=dense_mass,
            inverse_mass_matrix=inverse_mass_matrix,
            target_accept_prob=0.95,
            max_tree_depth=11
            #regularize_mass_matrix=False # this speeds up sampling for unknown reason
            )

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)

rng_key = random.PRNGKey(0)
mcmc.run(rng_key, sample_keys, param_bounds, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))


mcmc.print_summary()


import dill
with open("1339_jnkep_nburn-1000_nsteps-1000_hessian_tree-11_accept-95_emax-0.5_infl_err6.pkl", "wb") as f:
    dill.dump(mcmc, f)


