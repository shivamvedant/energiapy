import pyomo.dae as dae
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from curb.constraints import *
from curb.plot import plot_results
import os

os.environ["PATH"] = r"C:\GAMS\45;" + os.environ["PATH"]

tf = 100
nfe = 100
deltaG_ads = 38593.79 # J/mol
P = 101325.0 * 1 # Pa
k = 0.01  # cm/s
R = 8.3145  # J/(mol K)
F = 96485.0  # C/mol
U_applied = -0.5
V_rxn = 1.0
T_sim = 298.15
CO2_conc_feed = P / R / T_sim * 1e-6  # mol/cm^3
electrode_surface_area = 4.0  # cm^2
max_current_density = 200.0 # mA/cm^2
species = ["CO2", "CO", "etOH", "acetate", "C2H4", "formate", "CH4", "meOH", "prOH", "H2", "O2", "OH-", "H+"]
cathode_products = ["CO", "etOH", "acetate", "C2H4", "formate", "CH4", "meOH", "prOH", "H2"]


# unitless
alpha_cat = {
    "CO": 0.5,
    "etOH": 0.5,
    "acetate": 0.5,
    "C2H4": 0.5,
    "formate": 0.5,
    "CH4": 0.5,
    "meOH": 0.5,
    "prOH": 0.5,
    "H2": 0.5
}

z = {
    "CO":       2,
    "etOH":     12,
    "acetate":  8,
    "C2H4":     12,
    "formate":  2,
    "CH4":      8,
    "meOH":     6,
    "prOH":     18,
    "H2":       2,
    "O2":       4
}

# units of V vs RHE
E0_eq = {
    "CO":      -0.10,
    "etOH":     0.09,
    "acetate":  0.11,
    "C2H4":     0.08,
    "formate": -0.20,
    "CH4":      0.17,
    "meOH":     0.03,
    "prOH":     0.10,
    "H2":       0.00,
    "O2":       0.82
}

# units of mA/cm^2
j0 = {
    "CO":      1.00e-12,
    "etOH":    1.00e-12,
    "acetate": 1.00e-12,
    "C2H4":    1.00e-12,
    "formate": 1.00e-12,
    "CH4":     1.00e-12,
    "meOH":    1.00e-12,
    "prOH":    1.00e-12,
    "H2":      1.00e-12,
    "O2":      1.00e-12
}


activity_exp = {
    "CO":      1,
    "etOH":    2,
    "acetate": 2,
    "C2H4":    2,
    "formate": 1,
    "CH4":     1,
    "meOH":    1,
    "prOH":    3,
    "H2":      0,
    "O2":      0
}

model = pyo.ConcreteModel()



model.species = pyo.Set(initialize = species)
model.cathode_products = pyo.Set(initialize = cathode_products)
model.anode_products = pyo.Set(initialize = anode_products)
model.products = pyo.Set(initialize = model.anode_products | model.cathode_products)
model.Q = pyo.Param(model.products, initialize = Q)
model.time = dae.ContinuousSet(bounds=(0,tf))
model.CO2_conc_feed = pyo.Param(initialize = CO2_conc_feed)
model.max_current_density = pyo.Param(initialize = max_current_density)
model.alpha_cat = pyo.Param(model.products, initialize = alpha_cat)
model.E_eq = pyo.Param(model.products, initialize = E0_eq)
model.j0 = pyo.Param(model.products, initialize = j0)
model.z = pyo.Param(model.products, initialize = z)
model.activity_exp = pyo.Param(model.products, initialize = activity_exp)
model.dG = pyo.Param(initialize = deltaG_ads)
model.R = pyo.Param(initialize = R)
model.SA = pyo.Param(initialize = electrode_surface_area)
model.F = pyo.Param(initialize = F)
model.V = pyo.Param(initialize = V_rxn)
model.k = pyo.Param(initialize = k)

model.pH = pyo.Var(model.time)
model.T = pyo.Var(model.time)
model.U = pyo.Var(model.time, initialize = U_applied)
model.E_eq_i = pyo.Var(model.time, model.products)
model.C = pyo.Var(model.time, model.species)
model.N_CO2 = pyo.Var(model.time)
model.K_ads = pyo.Var(model.time)
model.j = pyo.Var(model.time, model.products)
model.r_electro = pyo.Var(model.time, model.products)
model.theta = pyo.Var(model.time)
model.dCdt = dae.DerivativeVar(model.C, wrt = model.time)
model.FE = pyo.Var(model.time, model.products)

# temperary variables to track the current density calculation for each component
model.eta = pyo.Var(model.time, model.products)
model.theta_activity = pyo.Var(model.time, model.products)
model.exp_term = pyo.Var(model.time, model.products)

pyo.TransformationFactory('dae.finite_difference').apply_to(model, nfe=nfe, wrt=model.time, scheme='FORWARD')

for t in model.time:
    model.T[t].fix(T_sim)
    model.pH[t].fix(7.0)
    model.U[t].fix(U_applied)

# define constraints
model.mass_balance_ic = pyo.Constraint(model.species, rule=mass_balance_ic_rule)
model.co2_molar_flux = pyo.Constraint(model.time, rule=co2_absorption_molar_flux_rule)
model.co2_mass_balance = pyo.Constraint(model.time, rule=co2_mass_balance_rule)
model.ads_equilibrium = pyo.Constraint(model.time, rule=ads_equilibrium_rule)
model.theta_def = pyo.Constraint(model.time, rule=theta_rule)
model.nerst_eqn = pyo.Constraint(model.time, model.cathode_products, rule=nerst_equation_rule)
model.temp_eta = pyo.Constraint(model.time, model.cathode_products, rule=temp_eta_rule)
model.temp_theta_activity = pyo.Constraint(model.time, model.cathode_products, rule=temp_theta_activity_rule)
model.temp_exp_term = pyo.Constraint(model.time, model.cathode_products, rule=temp_exp_term_rule)
# model.rxn_rate_constr = pyo.Constraint(model.time, model.cathode_products, rule=rxn_rate_rule)
# model.cathode_current_density = pyo.Constraint(model.time, model.cathode_products, rule=cathode_current_density_rule)


# model.anode_current_density = pyo.Constraint(model.time, model.anode_products, rule=anode_current_density_rule)
# model.max_current_density_constr = pyo.Constraint(model.time, rule=max_current_density_rule)
# model.H_ion_conc = pyo.Constraint(model.time, rule=H_ion_concentration)
# model.H_OH_balance = pyo.Constraint(model.time, rule=H_OH_balance_rule)

# model.mass_balance = pyo.Constraint(model.time, model.cathode_products | model.anode_products, rule=mass_balance_rule)
# model.FE_definition = pyo.Constraint(model.time, model.products, rule=FE_rule)

# model.obj = pyo.Objective(expr = 0.0, sense = pyo.maximize)

# solver = pyo.SolverFactory('gams', solver='baron',warmstart=True, executable=r"C:\GAMS\45\gams.exe")
# solver = pyo.SolverFactory('ipopt', executable=r"C:\Users\daniel.ribeiro\.conda\envs\CURBModel_Env\Library\bin\ipopt.exe")

results = solver.solve(model,tee = True)
results.write()


