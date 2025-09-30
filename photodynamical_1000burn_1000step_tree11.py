import numpy as np
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import config, random
import numpyro, jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
config.update('jax_enable_x64', True)
numpyro.set_platform('cpu')
num_chains = 4
numpyro.set_host_device_count(num_chains)
print ('# jax device count:', jax.local_device_count())
import importlib_resources
from jnkepler.jaxttv.infer import *
from jnkepler.nbodytransit import *



d = pd.read_csv("toi_1339_transit_data_all_planets.txt", sep="\s+", header=0, names=['Planet_num', 'Index', 'Tc', 'Tc_err'])

tcobs = [jnp.array(d.Tc[d.Planet_num==j+1]) for j in range(3)]
errorobs = [jnp.array(d.Tc_err[d.Planet_num==j+1]) for j in range(3)]
p_init = [8.88020257, 28.58140011, 38.35180318]


t_start = 1715.  # start of integration
t_end = 3670.
dt = p_init[0] / 40. # integration timestep

# detrended light curves around transits; here simultaneous transit is excluded
dlc = pd.read_csv("TOI1339_lc_photodyn.csv")
time_obs = jnp.array(dlc['time'])
flux_obs = jnp.array(dlc['flux']) - 1.  # normalize around 0
error_obs = jnp.array(dlc['flux_err'])

# Get the sorting indices
sort_idx = np.argsort(time_obs)

# Apply the same reordering to all arrays
time_obs = time_obs[sort_idx]
flux_obs = flux_obs[sort_idx]
error_obs = error_obs[sort_idx]

window = 0.6  # days
time_segments, flux_segments, error_segments = [], [], []

planet_ids = []

for planet_idx, tplanet in enumerate(tcobs):
    for t0 in tplanet:
        mask = (time_obs > t0 - window) & (time_obs < t0 + window)
        time_segments.append(time_obs[mask])
        flux_segments.append(flux_obs[mask])
        error_segments.append(error_obs[mask])
        planet_ids.append(np.ones(np.sum(mask)) * planet_idx)

# Concatenate segments
time_obs_trim = np.concatenate(time_segments)
flux_obs_trim = np.concatenate(flux_segments)
error_obs_trim = np.concatenate(error_segments)
planet_ids = np.concatenate(planet_ids).astype(int)

time_obs = jnp.array(time_obs_trim)
flux_obs = jnp.array(flux_obs_trim)
error_obs = jnp.array(error_obs_trim)
planet_ids = jnp.array(planet_ids)  


############## bin
from scipy.stats import binned_statistic

kepler_bin = 29.4
tess_bin = 2.0
def bin_light_curve(time, flux, flux_err, bin_width=kepler_bin/1440.):
    """
    Bin the light curve by computing the weighted mean flux in each time bin.

    Parameters:
    - time: array-like, time values (unevenly spaced)
    - flux: array-like, flux measurements corresponding to time
    - flux_err: array-like, flux measurement errors
    - bin_width: float, width of each time bin (same units as time)

    Returns:
    - bin_centers: array of bin center times
    - binned_flux: array of weighted mean flux per bin
    - binned_err: array of error on weighted mean flux per bin
    - bin_counts: number of points per bin
    """

    # Define bin edges
    t_min, t_max = np.min(time), np.max(time)
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    # Which bin each point belongs to
    binnumber = np.digitize(time, bins) - 1
    nbins = len(bins) - 1

    # Weights from errors
    w = 1.0 / flux_err**2

    # Weighted sums
    sum_w  = np.bincount(binnumber, weights=w, minlength=nbins)
    sum_fw = np.bincount(binnumber, weights=flux * w, minlength=nbins)

    # Weighted mean and its error
    binned_flux = np.full(nbins, np.nan)
    binned_err  = np.full(nbins, np.nan)

    mask = sum_w > 0
    binned_flux[mask] = sum_fw[mask] / sum_w[mask]
    binned_err[mask]  = np.sqrt(1.0 / sum_w[mask])

    # Counts per bin
    bin_counts = np.bincount(binnumber, minlength=nbins)

    # Bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return bin_centers, binned_flux, binned_err, bin_counts

idx_bin = time_obs > 0 #2250
#plt.xlim(3480, 3482)
# plt.plot(time_obs[idx_bin], flux_obs[idx_bin], '.')
tbin, fbin, ferr, counts = bin_light_curve(time_obs[idx_bin], flux_obs[idx_bin], error_obs[idx_bin])
idxc = counts > 5
time_obs = np.r_[time_obs[~idx_bin], tbin[idxc]]
flux_obs = np.r_[flux_obs[~idx_bin], fbin[idxc]]
error_obs = np.r_[error_obs[~idx_bin], ferr[idxc]]
# plt.plot(time_obs, flux_obs, '.')


time_obs = jnp.array(time_obs)
flux_obs = jnp.array(flux_obs)
error_obs = jnp.array(error_obs)
# planet_ids = jnp.array(planet_binned)



nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=True)
nt.set_lcobs(time_obs) #, exposure_time=2./1440., supersample_factor=0, overlapping_transit=True)

param_bounds_ttv = ttv_default_parameter_bounds(nt)#, emax=0.5)

p_init = np.array([8.88024377, 28.58110534, 38.35139431,
                   0.018993513, -0.04089329, -0.03720727,
                   0.02671717, -0.03057353, -0.02621894,
                   1715.35568461, 1726.05376508, 1743.55531096,
                   -10.97162589, -11.45687482, -11.99588567,
])

popt = ttv_optim_curve_fit(nt, param_bounds_ttv, p_init=p_init)
print(popt)

tc, _ = nt.check_timing_precision(popt)

pdic_normal, pdic_student = nt.check_residuals(popt)

##########################################################################################################
from jnkepler.infer import optim_svi
popt["smass"] = 0.81
popt["pmass"] = popt["pmass"] * popt["smass"]
n_planets = 3

keys_ttv = ["ecosw", "esinw", "pmass", "period", "tic"]
keys_transit = ["radius_ratio", "b", "srad", "q1", "q2", "meanflux"]
param_bounds_transit = {
    "q1": [jnp.array(0), jnp.array(1.)],
    "q2": [jnp.array(0), jnp.array(1.)],
    "radius_ratio":  [jnp.zeros(len(tcobs)), jnp.ones(len(tcobs))*0.1],
    "b": [jnp.zeros(len(tcobs)), jnp.ones(len(tcobs))],
    "srad": [jnp.array(0.93), jnp.array(0.95)], 
    "meanflux": [jnp.array(-5e-4), jnp.array(5e-4)]
}


import celerite2
from celerite2.jax import terms as jax_terms

def model_fix_ttv(par_dict, param_bounds_transit):
    par = {}

    for key in keys_ttv + ["smass"]:
        par[key] = par_dict[key]

    for key in keys_transit:
        if key in ["q1", "q2", "b"]:
            par[key] = numpyro.sample(key, dist.Uniform(param_bounds_transit[key][0], param_bounds_transit[key][1]))
        else:
            par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds_transit[key][0]*0, param_bounds_transit[key][0]*0+1.))
            par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds_transit[key][1] - param_bounds_transit[key][0]) + param_bounds_transit[key][0])

    
    fluxmodel, tcmodel = nt.get_flux(par)
    numpyro.deterministic("tcmodel", tcmodel)
    fluxmodel = numpyro.deterministic("fluxmodel", fluxmodel)    
    residual = numpyro.deterministic("residual", flux_obs - fluxmodel - par["meanflux"])
    numpyro.deterministic("normed_residual", residual / error_obs)
    
    # GP likelihood
    lna = numpyro.sample("lna", dist.Uniform(low=-14, high=-4))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
    lnjitter = numpyro.sample("lnjitter", dist.Uniform(low=-14, high=-4))
    jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(nt.times_lc, diag=error_obs**2 + jitter**2)
    numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    numpyro.deterministic("gppred", gp.predict(residual))


popt_transit = optim_svi(model_fix_ttv, 1e-2, 5000, par_dict=popt, param_bounds_transit=param_bounds_transit)

###################################################################################################################
popt_ttv_scaled = scale_pdic(popt, param_bounds_ttv)
popt_full = popt_transit.copy()
popt_full.update(popt_ttv_scaled)

popt_full['lnode_inner'] = jnp.zeros(2)


def model_full(param_bounds_ttv, param_bounds_transit):
    """full photodynamical model
    
        - noise model is iid gaussian; we could've used other noise model (e.g., GP, Student's t)
        - we could've fitted mean flux as well
        - stellar mass is fixed to 1Msun and only radius is floated, because only mean stellar density is constrained from this model;
          so pmass is mass ratio and stellar radius (srad) is in units of Rsun/(Mstar/Msun)**(1./3.)
        - impact parameters are restricted to be positive but this could be relaxed
        - longitude of ascending node (lnode) is measured with respect to the outermost transiting planet

    """
    par = {}

    par["smass"] = numpyro.sample("smass", dist.Normal(0.81, 0.04))

    # TTV parameters
    for key in keys_ttv:
        par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds_ttv[key][0]*0, param_bounds_ttv[key][0]*0+1.))
        par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds_ttv[key][1] - param_bounds_ttv[key][0]) + param_bounds_ttv[key][0])

    # add longitude of ascending node; prior Unif(-1, 1)
    par["lnode"] = numpyro.deterministic("lnode", jnp.zeros(2))
    # Jacobian for uniform ecc prior
    ecc = numpyro.deterministic("ecc", jnp.sqrt(par['ecosw']**2+par['esinw']**2))
    numpyro.factor("eprior", -jnp.log(ecc))

    # transit parameters (though note that b is used to fix inclination along with other TTV parameters)
    for key in keys_transit:
        if key in ["q1", "q2", "b"]:
            par[key] = numpyro.sample(key, dist.Uniform(param_bounds_transit[key][0], param_bounds_transit[key][1]))
        elif key == "srad":
            par[key] = numpyro.sample(key, dist.Normal(0.94, 0.02))
        else:
            par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds_transit[key][0]*0, param_bounds_transit[key][0]*0+1.))
            par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds_transit[key][1] - param_bounds_transit[key][0]) + param_bounds_transit[key][0])
    
    # compute model (tcmodel is not used)
    fluxmodel, tcmodel = nt.get_flux(par)
    numpyro.deterministic("tcmodel", tcmodel)
    numpyro.deterministic("fluxmodel", fluxmodel)  

    # GP likelihood  
    residual = numpyro.deterministic("residual", flux_obs - fluxmodel - par["meanflux"])
    numpyro.deterministic("normed_residual", residual / error_obs)
    lna = numpyro.sample("lna", dist.Uniform(low=-14, high=-4))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
    lnjitter = numpyro.sample("lnjitter", dist.Uniform(low=-14, high=-4))
    jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(nt.times_lc, diag=error_obs**2 + jitter**2)
    numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    numpyro.deterministic("gppred", gp.predict(residual))

    
popt_full = optim_svi(model_full, 1e-2, 5000, 
                      p_initial=popt_full, 
                      param_bounds_ttv=param_bounds_ttv, 
                      param_bounds_transit=param_bounds_transit)

import pickle 
with open('1339_jnkep_initial_fit_photodyn_tree11.pkl', 'wb') as f:
    pickle.dump({'nt': nt, 
                 'popt_transit': popt_transit, 
                 'popt_full': popt_full, 
                 'param_bounds': param_bounds_ttv, 
                 'param_bounds_transit': param_bounds_transit, 
                 'keys_ttv': keys_ttv, 
                 'keys_transit': keys_transit}, f)

print('Initial fit data saved successfully')


from numpyro.infer import MCMC, NUTS
kernel = NUTS(model_full, dense_mass=True, init_strategy=init_to_value(values=popt_full), max_tree_depth=11)

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)

from jax import random
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, param_bounds_ttv, param_bounds_transit, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))

mcmc.print_summary()

# save output
import dill
with open("toi1339_photodynamics_1000burn_1000step_tree11.pkl", "wb") as f:
    dill.dump(mcmc, f)