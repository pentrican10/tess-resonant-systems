
import numpy as np 
import rebound
import os
import matplotlib.pyplot as plt
import dill
import pandas as pd

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
####################################################################################

### get the posterior data
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

##############################################################################################################
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

Mean_anomaly_c = mean_anomaly(ref_time, omega_c, e_c, Tc_c, P_c)
Mean_anomaly_d = mean_anomaly(ref_time, omega_d, e_d, Tc_d, P_d)

### keep Omega constant
value_Omega_c = 0  # [deg]
Omega_c = np.full_like(omega_c, np.radians(value_Omega_c))
value_Omega_d = 0 # [deg]
Omega_d = np.full_like(omega_d, np.radians(value_Omega_d))

pomega_c = omega_c + Omega_c
pomega_d = omega_d + Omega_d
lambda_c = omega_c + Omega_c + Mean_anomaly_c
lambda_d = omega_d + Omega_d + Mean_anomaly_d

# Set fixed inclination (e.g. edge-on)
inc_c = [np.pi / 2] * len(a_c)
inc_d = [np.pi / 2] * len(a_d)


r_apo_c = a_c * (1 + e_c)
r_peri_d = a_d * (1 - e_d)
initially_crossed=[]
for i in range(len(r_apo_c)):
    r_apo = r_apo_c[i]
    r_peri = r_peri_d[i]
    if r_peri < r_apo:
        initially_crossed.append(i)

print(f'Systems where initial posterior state has r_peri_d < r_apo_c: {len(initially_crossed)}')

# Set parameters (fixed values from literature) median values from orell-miquel
from astropy.constants import M_jup, M_sun
mass_b = 0.3530 * (M_jup / M_sun)
a_b = 0.0804
e_b = 0.031
omega_b = np.radians(5.)
Omega_b = 0.0 # assume 0
inc_b = np.radians(88.10) #np.pi / 2
Tc_b = 2459443.54236 - tess_offset
P_b = 8.8803256



mass_e = 0.3590 * (M_jup / M_sun)  # MJup to Msun
a_e = 0.407
e_e = 0.031
omega_e = np.radians(-130.0)
Omega_e = 0.0  # Assume 0 for simplicity
inc_e = np.pi / 2  # edge-on
Tc_e = 2459348.12 - tess_offset
P_e = 101.12

# Reference time = same as for other planets
# t_ref = times[i]  # or pick a fixed time like 2459000

M_e = mean_anomaly(ref_time, omega_e, e_e, Tc_e, P_e)
M_b = mean_anomaly(ref_time, omega_b, e_b, Tc_b, P_b)


#################################################################################################################


def simulate_draw(i, m_star, a_c, a_d, m_c, m_d, e_c, e_d, inc_c, inc_d,
                  Omega_c, Omega_d, omega_c, omega_d, M_c, M_d,
                  tmax=1e6):
    """
    Simulate a single posterior draw for two planets and return True if stable.
    Assumes units: AU, Msun, days.
    dt in years?
    """
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
            e=e_c[i],
            inc=inc_c[i],
            Omega=Omega_c[i],
            omega=omega_c[i],
            M=M_c[i])

    # Planet d
    sim.add(m=m_d[i],
            a=a_d[i],
            e=e_d[i],
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
    sim.integrator = "whfast"
    dt = P_b/10
    sim.dt = dt

    # Optional: encounter detection
    r_hill = a_c[i] * np.sqrt(m_c[i] / (3*m_star))
    sim.exit_min_distance = r_hill  # [AU], or adjust to Roche limit

    try:
        Noutputs = 90000 # number of points along orbit

        # arrays to store positions
        Nplanets = len(sim.particles) - 1  # skip star
        x = np.zeros((Noutputs, Nplanets))
        y = np.zeros((Noutputs, Nplanets))
        z = np.zeros((Noutputs, Nplanets))
        e = np.zeros((Noutputs, Nplanets))
        a = np.zeros((Noutputs, Nplanets))
        inc = np.zeros((Noutputs, Nplanets))
        # choose times: 0 → tmax
        times = np.linspace(0, tmax, Noutputs)

        # dt_output = tmax / Noutputs
        for k, t in enumerate(times):
            # sim.integrate(sim.t + dt_output)
            sim.integrate(t, exact_finish_time=0)
            for j in range(Nplanets):
                p = sim.particles[j+1]  # skip star at index 0
                x[k,j] = p.x
                y[k,j] = p.y
                z[k,j] = p.z 
                e[k,j] = p.e
                a[k,j] = p.a
                inc[k,j] = p.inc
                
        # sim.integrate(tmax)

        
        if (e[:,0] >= 1.0).any() or (e[:,1] >= 1.0).any() or (e[:,2]>=1.0).any() or (e[:,3]>= 1.0).any():
            note = "escape or unbound"
            print(note)
            return False, note, sim, x, y, z, e, a, inc

        cross_times = []
        for k in range(Noutputs):
            Q_c = a[k,1] * (1 + e[k,1]) # apo for inner planet 
            q_d = a[k,2] * (1 - e[k,2]) # peri for outer planet
            
            if q_d < Q_c:
                cross_times.append(times[k])
                # note = "orbit crossing"
                # print(note)
                # return False, note, sim, x, y, z, e, a, inc
        if cross_times:
            note = "orbit crossing"
            print(note)
            return False, note, sim, x, y, z, e, a, inc
        else:        
            note='stable'
            print(note)
            return True, note, sim, x, y, z, e, a, inc

    except rebound.Encounter:
        note = f"encounter"
        print(note)
        return False, note, sim, x, y, z, e, a, inc
    

############################################################################################################

n_draws = 2  # or whatever number feels fast enough
N_total = len(P_c)  # assuming all arrays are same length

# Choose `n_draws` unique random indices
random_indices = np.random.choice(N_total, size=n_draws, replace=False)




tmax = 1e8 * np.median(P_c)
print(tmax)
draw_idx = []
results = []
notes = []
sims = []

master_data = []
sim_dir = "sim_details_8_chain"
os.makedirs(sim_dir, exist_ok=True)

for i in random_indices:
    stable, note, sim, x, y, z, e, a, inc = simulate_draw(
        i=i,
        m_star=m_star,
        a_c=a_c, a_d=a_d,
        m_c=m_c, m_d=m_d,
        e_c=e_c, e_d=e_d,
        inc_c=inc_c, inc_d=inc_d,
        Omega_c=Omega_c, Omega_d=Omega_d,
        omega_c=omega_c, omega_d=omega_d,
        M_c=Mean_anomaly_c, M_d=Mean_anomaly_d,
        tmax = tmax
    )
    draw_idx.append(i)
    results.append(stable)
    notes.append(note)
    sims.append(sim)
    # rebound.OrbitPlotSet(sim)
    print(a)
    print(e)

    # SAVE DATA
    # Create per-simulation dataframe for integration steps
    times = np.linspace(0, tmax, len(e))  # len(e) is number of timesteps
    step_df = pd.DataFrame({"time_days": times})
    
    Nplanets = e.shape[1]  # number of planets
    
    for p in range(Nplanets):
        step_df[f"x_p{p+1}"] = x[:, p]
        step_df[f"y_p{p+1}"] = y[:, p]
        step_df[f"z_p{p+1}"] = z[:, p]
        step_df[f"e_p{p+1}"] = e[:, p]
        step_df[f"a_p{p+1}"] = a[:, p]
        step_df[f"inc_p{p+1}"] = inc[:, p]
    
    # Save per-simulation dataframe
    step_filename = f"{sim_dir}/sim_{i}.csv"
    step_df.to_csv(step_filename, index=False)

    # Add a row to master dataframe list
    master_data.append({
        "draw_index": i,
        "stable": stable,
        "note": note,
        "m_b": mass_b, "a_b": a_b, "e_b": e_b, "inc_b": inc_b, "Omega_b": Omega_b, "omega_b": omega_b, "M_b": M_b,
        "m_c": m_c[i], "a_c": a_c[i], "e_c": e_c[i], "inc_c": inc_c[i], "Omega_c": Omega_c[i], "omega_c": omega_c[i], "M_c": Mean_anomaly_c[i],
        "m_d": m_d[i], "a_d": a_d[i], "e_d": e_d[i], "inc_d": inc_d[i], "Omega_d": Omega_d[i], "omega_d": omega_d[i], "M_d": Mean_anomaly_d[i],
        "m_e": mass_e, "a_e": a_e, "e_e": e_e, "inc_e": inc_e, "Omega_e": Omega_e, "omega_e": omega_e, "M_e": M_e,
        "detail_file": step_filename
    })
    



    # # PLOTS
    # # orbit plot
    # plt.figure(figsize=(8,8))
    # Nplanets = len(sim.particles) - 1  # skip star
    # for j in range(Nplanets):
    #     plt.plot(x[:,j], z[:,j], '.', markersize=1, label=f'Planet {j+1}')
    # plt.plot(0,0,'o', color='orange', label='Star')
    # plt.gca().set_aspect('equal', 'box')
    # plt.xlabel('x [AU]')
    # plt.ylabel('y [AU]')
    # plt.legend()
    # plt.show()

    # # eccentricity plot 
    # # --- Eccentricity evolution plot ---
    # plt.figure(figsize=(10,5))
    # times = np.linspace(0, tmax, len(e))
    # for j in range(Nplanets):
    #     plt.plot(times, e[:,j], '.', markersize=1, label=f'Planet {j+1}')
    # plt.xlabel('Time [days]')
    # plt.ylabel('Eccentricity')
    # plt.title(f"Draw {i}: Eccentricity Evolution")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # # plt.figure(figsize=(10,5))
    # peri_c = a[:,1] * (1 - e[:,1])
    # apo_c = a[:,1] * (1 + e[:,1])
    # peri_d = a[:,2] * (1 - e[:,2])
    # apo_d = a[:,2] * (1 + e[:,2])
   

    # plt.figure(figsize=(10,5))
    # plt.plot(times, a[:,1], color='black', label=f'planet c semi-major axis')
    # plt.plot(times, a[:,2], color='grey', label=f'planet d semi-major axis')
    # # plt.plot(times, e[:,1], '.', markersize=1, color='orange', label=f'planet c')
    # # plt.plot(times, e[:,2], '.', markersize=1, color='green', label=f'planet d')
    # plt.plot(times, peri_c, 's', markersize=1, color='orange', alpha=0.5, label=f'Peri & apo for c')
    # plt.plot(times, apo_c, 's', markersize=1, color='orange', alpha=0.5)#,  label=f'Apo c')
    # plt.plot(times, peri_d, 's', markersize=1, color='green', alpha=0.5, label=f'Peri & apo for d')
    # plt.plot(times, apo_d, 's', markersize=1, color='green', alpha=0.5)#,  label=f'Apo d')

    # # Fill between same-colored lines
    # plt.fill_between(times, peri_c, apo_c, color='orange', alpha=0.2)  # Green shaded area
    # plt.fill_between(times, peri_d, apo_d, color='green', alpha=0.2)   # Orange shaded area

    # plt.xlabel('Time [days]')
    # plt.ylabel('semi-major axis')
    # plt.title(f"Draw {i}: apo and peri Evolution")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print(apo_c)
    # print(peri_d)
        



master_df = pd.DataFrame(master_data)
master_file_name = "simulation_master_8_chain.csv"
master_df.to_csv(master_file_name, index=False)
print(f"Saved master summary to {master_file_name} with {len(master_df)} simulations")


print(results)
print(notes)
print(draw_idx)


stable_count = sum(results)
total = len(results)
stable_fraction = stable_count / total
print(f"Fraction of stable posterior draws: {stable_fraction:.2f}")
print(f"{stable_count} stable of {total} total systems")


true_indices = [i for i, val in enumerate(results) if val]

draws = [draw_idx[i] for i in true_indices]
for draw in draws:
    m2=m_c[draw]
    a2=a_c[draw]
    e2=e_c[draw]
    inc2=inc_c[draw]
    Omega2=Omega_c[draw]            
    omega2=omega_c[draw]
    M2=Mean_anomaly_c[draw]
    apo_2 = a2*(1+e2)

    # Planet d
    m3=m_d[draw]
    a3=a_d[draw]
    e3=e_d[draw]
    inc3=inc_d[draw]
    Omega3=Omega_d[draw]
    omega3=omega_d[draw]
    M3=Mean_anomaly_d[draw]
    peri_3 = a3*(1+e3)

    print(f'ecc 2: {e2}')
    print(f'ecc 3: {e3}')
    print(f'apo 2: {apo_2}')
    print(f'peri 3: {peri_3}')
    print('\n')


# all orbit draw eccentricities 
for draw in draw_idx:
    m2=m_c[draw]
    a2=a_c[draw]
    e2=e_c[draw]
    inc2=inc_c[draw]
    Omega2=Omega_c[draw]            
    omega2=omega_c[draw]
    M2=Mean_anomaly_c[draw]
    apo_2 = a2*(1+e2)

    # Planet d
    m3=m_d[draw]
    a3=a_d[draw]
    e3=e_d[draw]
    inc3=inc_d[draw]
    Omega3=Omega_d[draw]
    omega3=omega_d[draw]
    M3=Mean_anomaly_d[draw]
    peri_3 = a3*(1-e3)

    print(f'ecc 2: {e2}')
    print(f'ecc 3: {e3}')
    print(f'apo 2: {apo_2}')
    print(f'peri 3: {peri_3}')
    print('\n')





