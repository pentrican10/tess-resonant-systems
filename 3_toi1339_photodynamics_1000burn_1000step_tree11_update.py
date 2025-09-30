import jax, os
jax_version = jax.__version__
major, minor, patch = (int(x) for x in jax_version.split(".")[:3])
if (major, minor, patch) >= (0, 4, 32):
    print(f"JAX version: {jax_version}")
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import numpy as np
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


from jnkepler.jaxttv.infer import *
from jnkepler.nbodytransit import *
import importlib_resources
path = importlib_resources.files('jnkepler').joinpath('data')  # path for test data

import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='times')
plt.rcParams["figure.figsize"] = (12,6)
from matplotlib import rc

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


### only include light curve around transits 
time_segments, flux_segments, error_segments = [], [], []

planet_ids = []

# from Lubin 2022
T14_b = 3.1 * 0.0416667
T14_c = 4.5 * 0.0416667
T14_d = 5.5 * 0.0416667

for planet_idx, tplanet in enumerate(tcobs):
    for t0 in tplanet:
        ### create mask window based on transit duration
        if (planet_idx ==0):
            window = 1.5 * (T14_b/2)
        elif (planet_idx ==1):
            window = 1.5 * (T14_c/2)
        elif (planet_idx==2):
            window = 1.5* (T14_d/2)
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

### bin light curve 
from scipy.stats import binned_statistic
kepler_bin = 29.4/1440.
tess_bin = 2.0/1440.
def bin_light_curve(time, flux, flux_err, bin_width=tess_bin):
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


idx_bin = time_obs > 2250
#plt.xlim(3480, 3482)
plt.plot(time_obs[idx_bin], flux_obs[idx_bin], '.')
tbin, fbin, ferr, counts = bin_light_curve(time_obs[idx_bin], flux_obs[idx_bin], error_obs[idx_bin])
idxc = counts > 5
idxc = counts > 2

# planet ids 
planet_bin = []
bin_width = tess_bin
binnumber = np.digitize(time_obs[idx_bin], 
                        np.arange(tbin[0]-bin_width/2, tbin[-1]+bin_width/2, bin_width)) - 1
for i in range(len(tbin)):
    mask = binnumber == i
    if np.any(mask):
        # majority vote for planet id inside this bin
        pid = np.bincount(planet_ids[idx_bin][mask]).argmax()
        planet_bin.append(pid)
    else:
        planet_bin.append(-1)  # placeholder for empty bin

planet_bin = np.array(planet_bin)


planet_ids = np.r_[planet_ids[~idx_bin], planet_bin[idxc]]


time_obs = np.r_[time_obs[~idx_bin], tbin[idxc]]
flux_obs = np.r_[flux_obs[~idx_bin], fbin[idxc]]
error_obs = np.r_[error_obs[~idx_bin], ferr[idxc]]


# make a common sort index
sort_idx = np.argsort(time_obs)

# apply to all per-time arrays
time_obs   = time_obs[sort_idx]
flux_obs   = flux_obs[sort_idx]
error_obs  = error_obs[sort_idx]
planet_ids = planet_ids[sort_idx]

# back to jax arrays if not already
time_obs   = jnp.array(time_obs)
flux_obs   = jnp.array(flux_obs)
error_obs  = jnp.array(error_obs)
planet_ids = jnp.array(planet_ids)


nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=True)
nt.set_lcobs(time_obs, exposure_time=tess_bin, supersample_factor=0, overlapping_transit=False)

param_bounds_ttv = ttv_default_parameter_bounds(nt)

popt = ttv_optim_curve_fit(nt, param_bounds_ttv)
print(popt)

tc, _ = nt.check_timing_precision(popt)

pdic_normal, pdic_student = nt.check_residuals(popt)

from jnkepler.infer import optim_svi
popt["smass"] = 0.81

keys_ttv = ["ecosw", "esinw", "pmass", "period", "tic"]
keys_transit = ["radius_ratio", "b", "srad", "q1", "q2", "meanflux"]
param_bounds_transit = {
    "q1": [jnp.array(0), jnp.array(1.)],
    "q2": [jnp.array(0), jnp.array(1.)],
    "radius_ratio":  [jnp.zeros(3), jnp.ones(3)*0.1],
    "b": [jnp.zeros(3), jnp.ones(3)*1.1],
    "srad": [jnp.array(0.89), jnp.array(0.99)],
    "meanflux": [jnp.array(-1e-4), jnp.array(1e-4)]
}

import tinygp
from tinygp.kernels import quasisep as qk

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
    numpyro.deterministic("fluxmodel", fluxmodel)    
    residual = numpyro.deterministic("residual", flux_obs - fluxmodel - par["meanflux"])
    numpyro.deterministic("normed_residual", residual / error_obs)
    
    # GP likelihood
    lna = numpyro.sample("lna", dist.Uniform(low=-14, high=-4))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
    lnjitter = numpyro.sample("lnjitter", dist.Uniform(low=-14, high=-4))
    jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))
    kernel = qk.Matern32(sigma=jnp.exp(lna), scale=jnp.exp(lnc))
    gp = tinygp.GaussianProcess(kernel, nt.times_lc, diag=error_obs**2 + jitter**2, mean=0.0)
    numpyro.deterministic("logprob", gp.log_probability(residual))
    numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    numpyro.deterministic("gppred", gp.predict(residual))

popt_transit = optim_svi(model_fix_ttv, 1e-2, 5000, par_dict=popt, param_bounds_transit=param_bounds_transit)

popt_ttv_scaled = scale_pdic(popt, param_bounds_ttv)
popt_full = popt_transit.copy()
popt_full.update(popt_ttv_scaled)

popt_full['lnode_inner'] = jnp.zeros(2)

def model_full(param_bounds_ttv, param_bounds_transit, eps=1e-4):
    """full photodynamical model
        - stellar mass is fixed to 1Msun and only radius is floated, because only mean stellar density is constrained from this model;
          so pmass is mass ratio and stellar radius (srad) is in units of Rsun/(Mstar/Msun)**(1./3.)
        - impact parameters are restricted to be positive but this could be relaxed
        - longitude of ascending node (lnode) is measured with respect to the outermost transiting planet
    """
    # fix stellar mass to 1Msun without losing generality
    par = {"smass": 1.}

    # TTV parameters
    for key in keys_ttv:
        par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds_ttv[key][0]*0, param_bounds_ttv[key][0]*0+1.))
        par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds_ttv[key][1] - param_bounds_ttv[key][0]) + param_bounds_ttv[key][0])

    # add longitude of ascending node; prior Unif(-1, 1)
    lnode_inner = numpyro.sample("lnode_inner", dist.Uniform(-jnp.ones(2), jnp.ones(2)))
    par["lnode"] = numpyro.deterministic("lnode", jnp.hstack([lnode_inner, 0.]))

    # Jacobian for uniform ecc prior
    ecc = numpyro.deterministic("ecc", jnp.sqrt(par['ecosw']**2+par['esinw']**2))
    numpyro.factor("eprior", -jnp.log(ecc + eps))

    # transit parameters (note though that b is used to fix inclination along with other TTV parameters)
    for key in keys_transit:
        if key in ["q1", "q2", "b"]:
            par[key] = numpyro.sample(key, dist.Uniform(param_bounds_transit[key][0], param_bounds_transit[key][1]))
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
    kernel = qk.Matern32(sigma=jnp.exp(lna), scale=jnp.exp(lnc))
    gp = tinygp.GaussianProcess(kernel, nt.times_lc, diag=error_obs**2 + jitter**2, mean=0.0)
    numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    numpyro.deterministic("gppred", gp.predict(residual))

popt_full = optim_svi(model_full, 1e-2, 5000, p_initial=popt_full, param_bounds_ttv=param_bounds_ttv, param_bounds_transit=param_bounds_transit)


import pickle 
with open('toi1339_photodynamics_initial_fit_tree11_update.pkl', 'wb') as f:
    pickle.dump({'nt': nt, 
                 'popt_transit': popt_transit, 
                 'popt_full': popt_full, 
                 'param_bounds': param_bounds_ttv, 
                 'param_bounds_transit': param_bounds_transit, 
                 'keys_ttv': keys_ttv, 
                 'keys_transit': keys_transit,
                 'time_obs': time_obs, 
                 'flux_obs': flux_obs}, f)

print('Initial fit data saved successfully')

from numpyro.infer import MCMC, NUTS
kernel = NUTS(model_full, dense_mass=True, init_strategy=init_to_value(values=popt_full), max_tree_depth=11)

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)

rng_key = random.PRNGKey(0)
mcmc.run(rng_key, param_bounds_ttv, param_bounds_transit, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))

mcmc.print_summary()

# save output
import dill
with open("toi1339_photodynamics_1000burn_1000step_tree11_update.pkl", "wb") as f:
    dill.dump(mcmc, f)

