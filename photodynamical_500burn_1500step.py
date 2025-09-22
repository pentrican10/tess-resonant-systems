import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import config, random
import numpyro, jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
config.update('jax_enable_x64', False)
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
dt = 0.1 # integration timestep

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




bin_width = 29.4 / 1440.0  # days (~0.02042 d, 29.4 min)

def bin_segment(t, f, e, pid, bin_width):
    """Bin one transit segment to bin_width (days)."""
    edges = np.arange(t.min(), t.max() + bin_width, bin_width)
    t_b, f_b, e_b, pid_b = [], [], [], []
    for i in range(len(edges) - 1):
        mask = (t >= edges[i]) & (t < edges[i+1])
        if not np.any(mask):
            continue
        w = 1.0 / e[mask]**2
        t_b.append(np.average(t[mask], weights=w))
        f_b.append(np.average(f[mask], weights=w))
        e_b.append(np.sqrt(1.0 / np.sum(w)))
        pid_b.append(pid)  # pid is a scalar, same for the whole segment
    return np.array(t_b), np.array(f_b), np.array(e_b), np.array(pid_b)

# Loop over all segments
binned = [bin_segment(t, f, e, pid[0] if np.ndim(pid) else pid, bin_width)
          for t, f, e, pid in zip(time_segments, flux_segments, error_segments, planet_ids)]

# Concatenate results
time_binned   = np.concatenate([b[0] for b in binned if len(b[0]) > 0])
flux_binned   = np.concatenate([b[1] for b in binned if len(b[1]) > 0])
error_binned  = np.concatenate([b[2] for b in binned if len(b[2]) > 0])
planet_binned = np.concatenate([b[3] for b in binned if len(b[3]) > 0])

# Save to dataframe
df_binned = pd.DataFrame({
    "time": time_binned,
    "flux": flux_binned,
    "flux_err": error_binned,
    "planet_number": planet_binned
})
df_binned.to_csv("toi1339_binned_lightcurves.csv", index=False)
print(f"Saved {len(df_binned)} binned points.")

time_obs = jnp.array(time_binned)
flux_obs = jnp.array(flux_binned)
error_obs = jnp.array(error_binned)
planet_ids = jnp.array(planet_binned)



nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=True)
nt.set_lcobs(time_obs)

param_bounds_ttv = ttv_default_parameter_bounds(nt)#, emax=0.5)

p_init = np.array([8.88020257, 28.58140011, 38.35180318,
                   0.16996202, -0.09855112, -0.0863744,
                   0.09494596, -0.04802669, -0.04114657,
                   1715.3554624, 1726.054591, 1743.55713087,
                   -10.90254937, -11.3564078 , -11.88909652,
])

popt = ttv_optim_curve_fit(nt, param_bounds_ttv)#, p_init=p_init)
print(popt)

tc, _ = nt.check_timing_precision(popt)

pdic_normal, pdic_student = nt.check_residuals(popt)

##########################################################################################################
from jnkepler.infer import optim_svi
popt["smass"] = 0.81

keys_ttv = ["ecosw", "esinw", "pmass", "period", "tic"]
keys_transit = ["radius_ratio", "b", "srad", "q1", "q2", "meanflux"]
param_bounds_transit = {
    "q1": [jnp.array(0), jnp.array(1.)],
    "q2": [jnp.array(0), jnp.array(1.)],
    "radius_ratio":  [jnp.ones(3)*0.005, jnp.ones(3)*0.08],
    "b": [jnp.zeros(3), jnp.ones(3)*0.9],
    "srad": [jnp.array(0.75), jnp.array(1.25)],
    "meanflux": [jnp.array(-1e-4), jnp.array(1e-4)]
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
    
    # # GP likelihood
    # lna = numpyro.sample("lna", dist.Uniform(low=-14, high=-4))
    # lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
    # lnjitter = numpyro.sample("lnjitter", dist.Uniform(low=-14, high=-4))
    # jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))
    # kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    # gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    # gp.compute(nt.times_lc, diag=error_obs**2 + jitter**2)
    # numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    # numpyro.deterministic("gppred", gp.predict(residual))

    # Simple iid Gaussian likelihood with jitter
    lnjitter = numpyro.sample("lnjitter", dist.Uniform(-14, -4).expand([3]))
    jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))

    # Assign planet-specific jitter to each datapoint
    point_jitter = numpyro.deterministic("point_jitter", jitter[planet_ids])
    
    # Likelihood
    numpyro.sample(
        "obs",
        dist.Normal(0.0, jnp.sqrt(error_obs**2 + point_jitter**2)),
        obs=residual,
    )


popt_transit = optim_svi(model_fix_ttv, 1e-2, 5000, 
                         par_dict=popt, 
                         param_bounds_transit=param_bounds_transit)

###################################################################################################################
popt_ttv_scaled = scale_pdic(popt, param_bounds_ttv)
popt_full = popt_transit.copy()
popt_full.update(popt_ttv_scaled)

popt_full['lnode_innder'] = jnp.zeros(2)


def model_full(param_bounds_ttv, param_bounds_transit):
    """full photodynamical model
    
        - noise model is iid gaussian; we could've used other noise model (e.g., GP, Student's t)
        - we could've fitted mean flux as well
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
    numpyro.factor("eprior", -jnp.log(ecc))

    # transit parameters (though note that b is used to fix inclination along with other TTV parameters)
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

    # # GP likelihood  
    residual = numpyro.deterministic("residual", flux_obs - fluxmodel - par["meanflux"])
    numpyro.deterministic("normed_residual", residual / error_obs)
    # lna = numpyro.sample("lna", dist.Uniform(low=-14, high=-4))
    # lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
    # lnjitter = numpyro.sample("lnjitter", dist.Uniform(low=-14, high=-4))
    # jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))
    
    # kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    # gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    # gp.compute(nt.times_lc, diag=error_obs**2 + jitter**2)
    
    # numpyro.sample("obs", gp.numpyro_dist(), obs=residual)
    # numpyro.deterministic("gppred", gp.predict(residual))

    # Simple iid Gaussian likelihood with jitter
    lnjitter = numpyro.sample("lnjitter", dist.Uniform(-14, -4).expand([3]))
    jitter = numpyro.deterministic("jitter", jnp.exp(lnjitter))

    # Assign planet-specific jitter to each datapoint
    point_jitter = numpyro.deterministic("point_jitter", jitter[planet_ids])
    
    # Likelihood
    numpyro.sample(
        "obs",
        dist.Normal(0.0, jnp.sqrt(error_obs**2 + point_jitter**2)),
        obs=residual,
    )

    
popt_full = optim_svi(model_full, 1e-2, 5000, 
                      p_initial=popt_full, 
                      param_bounds_ttv=param_bounds_ttv, 
                      param_bounds_transit=param_bounds_transit)

import pickle 
with open('1339_jnkep_initial_fit_photodyn.pkl', 'wb') as f:
    pickle.dump({'nt': nt, 
                 'popt_transit': popt_transit, 
                 'popt_full': popt_full, 
                 'param_bounds': param_bounds_ttv, 
                 'param_bounds_transit': param_bounds_transit, 
                 'keys_ttv': keys_ttv, 
                 'keys_transit': keys_transit}, f)

print('Initial fit data saved successfully')


from numpyro.infer import MCMC, NUTS
kernel = NUTS(model_full, dense_mass=True, init_strategy=init_to_value(values=popt_full), max_tree_depth=8)

mcmc = MCMC(kernel, num_warmup=500, num_samples=1500, num_chains=num_chains)

from jax import random
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, param_bounds_ttv, param_bounds_transit, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))

mcmc.print_summary()

# save output
import dill
with open("toi1339_photodynamics_500burn_1500step.pkl", "wb") as f:
    dill.dump(mcmc, f)