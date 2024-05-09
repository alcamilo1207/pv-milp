import numpy as np
from scipy import optimize as op

def getIdeal(energy_prices, solar_potential, max_energy_consuption,
             energy_requirement, verbose=False):


    energy_cost = energy_prices*max_energy_consuption  # C: Energy costs
    total_PV_energy = np.sum(solar_potential)
    usage = np.full_like(energy_cost,max_energy_consuption) # A: Energy usage = solar

    bounds = op.Bounds(0, 1)  # 0 <= x_i <= 1
    integrality = np.full_like(energy_cost, True)  # x_i are integers

    R = energy_requirement - total_PV_energy
    constraints = op.LinearConstraint(A=usage, lb=R, ub=R*1.2)

    res = op.milp(c=energy_cost, constraints=[constraints],
               integrality=integrality, bounds=bounds)

    if res.status == 0:
        used_energy = np.sum(usage*res.x) + total_PV_energy
        total_cost = np.sum(energy_cost*res.x)
        if verbose:
            print("Optimized energy consumption:", res.x)
            print("Total energy spent:", used_energy)
            print("Total cost:", total_cost)

        return res.x,total_cost
    else:
        return np.full_like(energy_prices,0),0
