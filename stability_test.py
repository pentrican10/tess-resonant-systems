import numpy as np 
import rebound
import os
import matplotlib.pyplot as plt
import dill

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



### get the posterior data
file_name = '1339_jnkep_fit_1339-2000burn-2000step-emax0.5.pkl'

working_dir = os.getcwd()
data_dir = os.path.join(working_dir, "data")

file_path = os.path.join(data_dir, file_name)

with open(file_name, "rb") as f:
    mcmc = dill.load(f)
mcmc.print_summary()
print(mcmc)

posterior = mcmc.get_samples(group_by_chain=True)  # shape: (num_chains, num_samples, ...)
print({k: v.shape for k, v in posterior.items()})


good_chain_idxs = [0,1,2]

samples_filtered = {
    k: v[good_chain_idxs, ...] for k, v in posterior.items()
}
# print(samples_filtered)
samples = {
    k: v.reshape(-1, *v.shape[2:]) for k, v in samples_filtered.items()
}

# Shape of samples
print({k: v.shape for k, v in samples.items()})



### conversions
solar_mass_2_kg = 1.989e30 # [kg per solar mass]
d_2_s = 86400 # [seconds per day]

### system constants 
m_star = 0.81  # [solar mass]

from astropy import units as u
from astropy import constants as c
G_const_si = c.G
G_const = c.G.to(u.AU**3 / u.M_sun / u.day**2).value


### posterior samples 
ecosw_b = samples["ecosw"][:, 0]
esinw_b = samples["esinw"][:, 0]
# e_b = samples["ecc"][:,0]
P_b = samples["period"][:,0] # [days]
Tc_b = samples["tic"][:,0] # [days]
m_b = samples["pmass"][:,0] # [solar mass]

ecosw_c = samples["ecosw"][:, 1]
esinw_c = samples["esinw"][:, 1]
# e_c = samples["ecc"][:,1]
P_c = samples["period"][:,1] # [days]
Tc_c = samples["tic"][:,1] # [days]
m_c = samples["pmass"][:,1] # [solar mass]  

### calculated params
a_b = np.array(((G_const*m_star*P_b**2)/(4*np.pi**2))**(1.0/3)) # [AU]
a_c = np.array(((G_const*m_star*P_c**2)/(4*np.pi**2))**(1/3)) # [AU]

e_b = np.sqrt(ecosw_b**2 + esinw_b**2)
e_c = np.sqrt(ecosw_c**2 + esinw_c**2)
omega_b = np.arctan2(esinw_b, ecosw_b) # [rad]
omega_c = np.arctan2(esinw_c, ecosw_c) # [rad]

# times = np.array([1980.] * len(omega_b))
times = Tc_c

Mean_anomaly_b = mean_anomaly(times, omega_b, e_b, Tc_b, P_b)
Mean_anomaly_c = mean_anomaly(times, omega_c, e_c, Tc_c, P_c)

### keep Omega constant
value_Omega_b = 0  # [deg]
Omega_b = np.full_like(omega_b, np.radians(value_Omega_b))
value_Omega_c = 0 # [deg]
Omega_c = np.full_like(omega_c, np.radians(value_Omega_c))

pomega_b = omega_b + Omega_b
pomega_c = omega_c + Omega_c
lambda_b = omega_b + Omega_b + Mean_anomaly_b
lambda_c = omega_c + Omega_c + Mean_anomaly_c

# Set fixed inclination (e.g. edge-on)
inc_b = [np.pi / 2] * len(a_b)
inc_c = [np.pi / 2] * len(a_c)


def simulate_draw(i, m_star, a_b, a_c, m_b, m_c, e_b, e_c, inc_b, inc_c,
                  Omega_b, Omega_c, omega_b, omega_c, M_b, M_c,
                  tmax=1e6, dt=0.05):
    """
    Simulate a single posterior draw for two planets and return True if stable.
    Assumes units: AU, Msun, days.
    """
    sim = rebound.Simulation()
    sim.units = ('AU', 'day', 'Msun')
    sim.G = G_const  # Set your custom G if using AU/day²/Msun

    # Central star
    sim.add(m=m_star)

    # Planet b
    sim.add(m=m_b[i],
            a=a_b[i],
            e=e_b[i],
            inc=inc_b[i],
            Omega=Omega_b[i],
            omega=omega_b[i],
            M=M_b[i])

    # Planet c
    sim.add(m=m_c[i],
            a=a_c[i],
            e=e_c[i],
            inc=inc_c[i],
            Omega=Omega_c[i],
            omega=omega_c[i],
            M=M_c[i])

    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = dt

    # Optional: encounter detection
    sim.exit_min_distance = 0.01  # [AU], or adjust to Roche limit

    try:
        sim.integrate(tmax)
        note='stable'
        print(note)
        return True, note

    # except Exception as e:
    #     note = f"Simulation {i} failed: {e}"
    #     print(note)
    #     return False, note
    except rebound.Encounter:
        note = f"encounter"
        print(note)
        return False, note
    except rebound.Escape:
        note = "escape"
        print(note)
        return False, note
    except Exception:
        note=f"exception: {Exception}"
        print(note)
        return False, note



n_draws = 200  # or whatever number feels fast enough
N_total = len(P_b)  # assuming all arrays are same length

# Choose `n_draws` unique random indices
random_indices = np.random.choice(N_total, size=n_draws, replace=False)


tmax = 1e6 * np.median(P_b)

results = []
notes = []
for i in random_indices:
    stable, note = simulate_draw(
        i=i,
        m_star=m_star,
        a_b=a_b, a_c=a_c,
        m_b=m_b, m_c=m_c,
        e_b=e_b, e_c=e_c,
        inc_b=inc_b, inc_c=inc_c,
        Omega_b=Omega_b, Omega_c=Omega_c,
        omega_b=omega_b, omega_c=omega_c,
        M_b=Mean_anomaly_b, M_c=Mean_anomaly_c,
        tmax = tmax
    )
    results.append(stable)
    notes.append(note)
print(results)
print(notes)

stable_count = sum(results)
total = len(results)
stable_fraction = stable_count / total
print(f"Fraction of stable posterior draws: {stable_fraction:.2f}")
print(f"{stable_count} stable of {total} total systems")