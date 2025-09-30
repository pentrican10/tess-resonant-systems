# tess-resonant-systems

## TOI 1339 analysis and results 

- Using updated version 0.2.1 jnkepler ([link](https://github.com/kemasuda/jnkepler))
- jupyter notebooks 0-8 contain the analysis and results for the transit time fitting and ttv analysis
- the mcmc run is run as a python file
    - python files starting with '3_' are meant to be run in order with the jupyter notebooks

- create conda environment using: **hd191939_env.yml** (fix batman)
- run order:
    - 0_tess_limb_darkening.ipynb
    - 1_transit_fit_linear.ipynb
    - 2_lithwick_omc.ipynb
    - 3_toi1339_photodynamics_1000burn_1000step_tree11_update.py
        - this is the primary file to run as of now, other .py files are also included in the repo
        - updated jnkepler, photodynamical model, data binned to 2-min, max tree depth=11
        - requires file with lightcurve data. The file is too large to put here, but can be recreated using lightkurve, or you can email pentrican10@g.ucla.edu for the direct file.
    - 4_toi_1339_run_analysis.ipynb
    - 5_stability_toi_1339.ipynb
    - 6_resonant_fig.ipynb
    - 7_resonant_plotting.ipynb
    - 8_e-w_constraint.ipynb
 
- complementary files: for testing and intermittent code for MAP model and mcmc runs
    - all files beginning with 'comp_'
 
- kep51 files are from the examples given in the [jnkepler](https://github.com/kemasuda/jnkepler) repository
