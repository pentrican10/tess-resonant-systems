import numpy as np 
import rebound
import os
import matplotlib.pyplot as plt
import dill
import pandas as pd


from spock import FeatureClassifier
from spock import DeepRegressor
from spock import AnalyticalClassifier
from spock import CollisionMergerClassifier

tess_offset =  2457000


def mean_anomaly(t, omega, e, Tc, P):
    """
    Calculate mean anomaly M at time t, given:
    - omega: argument of periastron (radians)
    - e: eccentricity
    - Tc: time of inferior conjunction (transit)
    - P: orbital period
    
    Converts Tc to periastron passage time tau using the formula:
    2*pi*(Tc - tau)/P = E0 - e*sin(E0)
    with E0 = 2 * arctan(sqrt((1-e)/(1+e)) * tan(pi/2 - omega/2))
    
    Returns mean anomaly M in radians [0, 2pi).
    """
    # Compute E0 from e and omega
    E0 = 2 * np.arctan( np.sqrt((1 - e) / (1 + e)) * np.tan(np.pi/4 - omega/2) )
    E0 = E0 % (2 * np.pi)

    # Compute time of periastron passage tau
    tau = Tc - (P / (2 * np.pi)) * (E0 - e * np.sin(E0))
    # print(tau[0], omega[0], e[0], Tc[0], P[0] )
    # print(f'tau: {tau[0]}') 
    # print(f'omega: {omega[0]}')
    # print(f'e: {e[0]}')
    # print(f'Tc: {Tc[0]}')
    # print(f'P: {P[0]}')

    # Compute mean anomaly at time t
    M = (2 * np.pi / P) * (t - tau)
    M = M % (2 * np.pi)

    return M

##########################################################################################################

file_name = '1339_jnkep_nburn-1000_nsteps-1000_hessian_tree-11_accept-95_emax-0.5.pkl'

working_dir = os.getcwd()
data_dir = os.path.join(working_dir, "data")

file_path = os.path.join(data_dir, file_name)



with open(file_name, "rb") as f:
    mcmc = dill.load(f)
mcmc.print_summary()
print(mcmc)

posterior = mcmc.get_samples(group_by_chain=True)  # shape: (num_chains, num_samples, ...)
print({k: v.shape for k, v in posterior.items()})

good_chain_idxs = [0,1,2,3,4,5,6,7]

samples_filtered = {
    k: v[good_chain_idxs, ...] for k, v in posterior.items()
}
# print(samples_filtered)
samples = {
    k: v.reshape(-1, *v.shape[2:]) for k, v in samples_filtered.items()
}

# Shape of samples
print({k: v.shape for k, v in samples.items()})

#############################################################################################################

### system constants 
m_star = 0.81  # [solar mass]

from astropy import units as u
from astropy import constants as c
G_const_si = c.G
G_const = c.G.to(u.AU**3 / u.M_sun / u.day**2).value


### posterior samples 
ecosw_c = samples["ecosw"][:, 0]
esinw_c = samples["esinw"][:, 0]
# e_c = samples["ecc"][:,0]
P_c = samples["period"][:,0] # [days]
Tc_c = samples["tic"][:,0] # [days]
m_c = samples["pmass"][:,0] # [solar mass]

ecosw_d = samples["ecosw"][:, 1]
esinw_d = samples["esinw"][:, 1]
# e_d = samples["ecc"][:,1]
P_d = samples["period"][:,1] # [days]
Tc_d = samples["tic"][:,1] # [days]
m_d = samples["pmass"][:,1] # [solar mass]  

### calculated params
a_c = np.array(((G_const*m_star*P_c**2)/(4*np.pi**2))**(1.0/3.)) # [AU]
a_d = np.array(((G_const*m_star*P_d**2)/(4*np.pi**2))**(1./3.)) # [AU]

e_c = np.sqrt(ecosw_c**2 + esinw_c**2)
e_d = np.sqrt(ecosw_d**2 + esinw_d**2)
omega_c = np.arctan2(esinw_c, ecosw_c) # [rad]
omega_d = np.arctan2(esinw_d, ecosw_d) # [rad]

# times = np.array([1980.] * len(omega_c))
times = Tc_d
ref_time = 0.0 # set to simulation start time

M_c = mean_anomaly(ref_time, omega_c, e_c, Tc_c, P_c)
M_d = mean_anomaly(ref_time, omega_d, e_d, Tc_d, P_d)

### keep Omega constant
value_Omega_c = 0  # [deg]
Omega_c = np.full_like(omega_c, np.radians(value_Omega_c))
value_Omega_d = 0 # [deg]
Omega_d = np.full_like(omega_d, np.radians(value_Omega_d))

pomega_c = omega_c + Omega_c
pomega_d = omega_d + Omega_d
lambda_c = omega_c + Omega_c + M_c
lambda_d = omega_d + Omega_d + M_d

# Set fixed inclination (e.g. edge-on)
inc_c = [0.]  * len(a_c)# [np.pi / 2] * len(a_c)
inc_d = [0.] * len(a_d)# [np.pi / 2] * len(a_d)


# Set parameters (fixed values from literature) Polanski, orell-miquel
from astropy.constants import M_jup, M_sun
mass_b = 0.0302 * (M_jup / M_sun)
a_b = 0.079
e_b = 0.0
omega_b = np.radians(90.)
Omega_b = 0.0 # assume 0
inc_b =0# np.pi / 2
Tc_b = 2458715.35572 - tess_offset
P_b = 8.8803232



mass_e = 0.3590 * (M_jup / M_sun)  # MJup to Msun
a_e = 0.400
e_e = 0.0
omega_e = np.radians(230.0)
Omega_e = 0.0  # Assume 0 for simplicity
inc_e = 0# np.pi / 2  # edge-on
Tc_e = 2459044.0 - tess_offset
P_e = 101.5

# Reference time = same as for other planets
# t_ref = times[i]  # or pick a fixed time like 2459000

M_e = mean_anomaly(np.median(Tc_d), omega_e, e_e, Tc_e, P_e)
M_b = mean_anomaly(np.median(Tc_d), omega_b, e_b, Tc_b, P_b)

pomega_b = omega_b + Omega_b
pomega_e = omega_e + Omega_e
lambda_b = omega_b + Omega_b + M_b
lambda_e = omega_e + Omega_e + M_e

#############################################################################################################


results = []

for i in range(len(P_c)):
    feature_model = FeatureClassifier()

    sim = rebound.Simulation()
    sim.units = ('AU', 'day', 'Msun')
    sim.G = G_const  # Set your custom G if using AU/day²/Msun
    
    # Central star
    sim.add(m=m_star)
    
    # Planet b
    sim.add(m=mass_b,
            a=a_b,
            e=e_b,
            inc=inc_b,
            Omega=Omega_b,
            omega=omega_b,
            M=M_b)
    
    # Planet c
    sim.add(m=m_c[i],
                a=a_c[i],
                e= e_c[i],
                inc=inc_c[i],
                Omega=Omega_c[i],
                omega= omega_c[i],
                M=M_c[i])
    
    # Planet d
    sim.add(m=m_d[i],
                a=a_d[i],
                e= e_d[i],
                inc=inc_d[i],
                Omega=Omega_d[i],
                omega=omega_d[i],
                M=M_d[i])
    
    # Planet e
    sim.add(m=mass_e,
            a=a_e,
            e=e_e,
            inc=inc_e,
            Omega=Omega_e,
            omega=omega_e,
            M=M_e)
    
        
    sim.move_to_com()
    # scalar probability of stability over a billion orbits
    stability_orbit_prob = feature_model.predict_stable(sim, Nbodytmax = 1e7)
    # print(f'prob of stability over 1e7 orbits of b: {stability_orbit_prob}')

    # estimate the median expected instability time 
    deep_model = DeepRegressor()
    deep_model_stability_prob = deep_model.predict_stable(sim)
    median, lower, upper, samples = deep_model.predict_instability_time(
        sim, samples=10000, return_samples=True, seed=0
    )
    expectation_log_norm = 10**np.average(np.log10(samples))
    # print(f'expectation of log-normal: {10**np.average(np.log10(samples))}')  # Expectation of log-normal    
    # print(f'median expected instability time: {median}')
    # fig, ax = plt.subplots()
    # ax.hist(np.log10(samples), density=True,
    #             histtype=u'step', bins=20,
    #             range=(4, 12), lw=3);
    
    # plt.ylabel('Probability')
    # plt.xlabel('Instability time (log10(T))')


    analytical_model = AnalyticalClassifier()
    analytical_stability_prob = analytical_model.predict_stable(sim)
    # 0 means confidently chaotic 
    # print(f'Probability of regular (non-chaotic) motion: {analytical_stability_prob}')

    # predicting collisional outcome
    class_model = CollisionMergerClassifier()
    
    prob_12, prob_23, prob_13 = class_model.predict_collision_probs(sim)
    
    # print(prob_12, prob_23, prob_13)


    ### save results 
    draw = i
    print(f'draw: {i}')
    
    # collect results
    run_result = {
        "draw": i,
        "params": {
            "m_c": m_c[i],
            "a_c": a_c[i],
            "e_c": e_c[i],
            "inc_c": inc_c[i],
            "Omega_c": Omega_c[i],
            "omega_c": omega_c[i],
            "M_c": M_c[i],
            "m_d": m_d[i],
            "a_d": a_d[i],
            "e_d": e_d[i],
            "inc_d": inc_d[i],
            "Omega_d": Omega_d[i],
            "omega_d": omega_d[i],
            "M_d": M_d[i],
        },
        "stability_prob": float(stability_orbit_prob),
        "dmodel_stability_prob": float(deep_model_stability_prob),
        "instability": {
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
            "expectation_log_normal": float(expectation_log_norm),
            # optional: store subset of samples (not all 10k if large)
            "samples": samples.tolist()[:100]  
        },
        "analytical_stability_prob": float(analytical_stability_prob),
        "collision_probs": {
            "prob_12": float(prob_12),
            "prob_23": float(prob_23),
            "prob_13": float(prob_13)
        }
    }

    results.append(run_result)

ext = '8_chain'
df = pd.json_normalize(results)
df.to_csv(f"spock_simulation_results_{ext}.csv", index=False)



