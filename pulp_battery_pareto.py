import pulp as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pvlib.irradiance as irrad
import pvlib.location as loc
mpl.use('Qt5Agg')


# load must equal supply for t in timesteps
    # supply = pv_gen_t + grid_from_t + battery_supply_t
    # load = load_total_t + grid_to_t + battery_load_t

# one of grid_from_t or grid_to_t must be 0

# battery soc at timestep t can't be more than soc_t-1 +- batt_max_rate
# battery soc at timestep t can't be more than batt_capacity
# battery soc at timestep t can't be less than 0
# one of battery_supply_t or battery_load_t must be 0

# cost = sum(grid_from_t * cost_grid_from_t - grid_to_t * rev_grid_to_t) for t in timesteps

# objective: chose valid values for grid_from_t, grid_to_t, battery_supply_t, battery_load_t such that cost is minimized

timesteps = 24
df = pd.DataFrame({
    'epex_spot_per_kwh': [150, 120, 110, 110, 110, 150, 190, 220, 220, 200, 160, 150, 150, 160, 170, 200, 220, 250, 250, 220, 200, 200, 190, 180],
    'emissions_per_kwh': [200, 200, 200, 180, 180, 180, 180, 180, 150, 150, 150, 100, 100, 100, 130, 150, 150, 150, 150, 150, 200, 200, 200, 200],
    'load_business': [10, 10, 10, 10, 10, 40, 40, 40, 50, 60, 100, 100, 50, 100, 100, 80, 50, 10, 10, 10, 40, 40, 40, 40],
    'load_homes': [10, 10, 10, 10, 10, 20, 60, 60, 80, 80, 20, 20, 20, 20, 20, 20, 40, 110, 110, 110, 50, 50, 30, 10]
}, index=pd.date_range('2022-06-01', freq='H', periods=timesteps))
df['load_homes'] *= 5
df['load_business'] *= 4
df['epex_spot_per_kwh'] /= 1000
df['load_total'] = df[['load_homes', 'load_business']].sum(axis=1)

l = loc.Location(52.091537, 5.107669)
cs = l.get_clearsky(df.index)
df['pv_gen'] = cs['ghi']
df['net_load'] = df['load_total'] - df['pv_gen']
df['emissions_per_kwh'] = df['emissions_per_kwh'].sub(df['emissions_per_kwh'].mean()).div(2).add(df['emissions_per_kwh'])
df.plot(subplots=True)


def hbm_lp(bat_cap=1000., bat_max_rate=700., bat_efficiency=.95, grid_max_rate=1000.):
    bat_t0_soc = .5 * bat_cap
    bat_tmax_soc = bat_t0_soc
    # define prob
    prob = pl.LpProblem("home_battery_management_system", pl.LpMinimize)

    # constants
    load_total = df['load_total'].values
    epex_spot = df['epex_spot_per_kwh'].values
    pv_gen = df['pv_gen'].values
    emissions_per_kwh = df['emissions_per_kwh'].values
    battery_efficiency = np.repeat(bat_efficiency, timesteps)

    # variables
    grid_from = [pl.LpVariable(f"grid_from_{t}", 0, grid_max_rate) for t in range(timesteps)]
    grid_to = [pl.LpVariable(f"grid_to_{t}", 0, grid_max_rate) for t in range(timesteps)]
    grid_consume = [pl.LpVariable(f"grid_consume_{t}", 0, 1, pl.LpBinary) for t in range(timesteps)]

    battery_from = [pl.LpVariable(f"battery_from_{t}", 0, bat_max_rate) for t in range(timesteps)]
    battery_to = [pl.LpVariable(f"battery_to_{t}", 0, bat_max_rate) for t in range(timesteps)]
    battery_consume = [pl.LpVariable(f"battery_consume_{t}", 0, 1, pl.LpBinary) for t in range(timesteps)]

    # constraints
    M_grid = 10*grid_max_rate
    M_battery = 10*bat_max_rate

    prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to, battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from, battery_efficiency)) == bat_tmax_soc, f'battery_soc_t0 is battery_soc_max'

    for t in range(timesteps):
        prob += pv_gen[t] + grid_from[t] + battery_from[t] == load_total[t] + grid_to[t] + battery_to[t], f'balance_{t}'

        prob += grid_from[t] - grid_consume[t]*M_grid <= 0, f'grid_from_directional_{t}'
        prob += grid_to[t] + grid_consume[t]*M_grid <= M_grid, f'grid_to_directional_{t}'
        prob += battery_from[t] - battery_consume[t] * M_battery <= 0, f'battery_from_directional_{t}'
        prob += battery_to[t] + battery_consume[t] * M_battery <= M_battery, f'battery_to_directional_{t}'

        prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to[:t+1], battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from[:t+1], battery_efficiency)) <= bat_cap, f'battery_max_soc_{t}'
        prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to[:t+1], battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from[:t+1], battery_efficiency)) >= 0, f'battery_min_soc_{t}'

    # objective
    prob += pl.lpSum(pl.lpDot(grid_from, epex_spot)) - pl.lpSum(pl.lpDot(grid_to, epex_spot))

    prob.solve()
    total_cost = pl.value(prob.objective)

    results = pd.DataFrame(dict(
        grid_from=[pl.value(grid_from[t]) for t in range(timesteps)],
        grid_to=[pl.value(grid_to[t]) for t in range(timesteps)],
        grid_consume=[pl.value(grid_consume[t]) for t in range(timesteps)],
        battery_from=[pl.value(battery_from[t]) for t in range(timesteps)],
        battery_to=[pl.value(battery_to[t]) for t in range(timesteps)],
        battery_consume=[pl.value(battery_consume[t]) for t in range(timesteps)]
    ), index=df.index)

    results['soc'] = bat_t0_soc + results['battery_to'].sub(results['battery_from']).cumsum()
    results['battery'] = results['battery_from'].sub(results['battery_to'])
    results['grid'] = results['grid_from'].sub(results['grid_to'])

    return total_cost, results


def hbm_lp_price_pareto(bat_cap=1000., bat_max_rate=700., bat_efficiency=.95, grid_max_rate=1000., price_pareto=None, price_interval_pareto=5.):
    bat_t0_soc = .5 * bat_cap
    bat_tmax_soc = bat_t0_soc
    # define prob
    prob = pl.LpProblem("home_battery_management_system", pl.LpMinimize)

    # constants
    load_total = df['load_total'].values
    epex_spot = df['epex_spot_per_kwh'].values
    pv_gen = df['pv_gen'].values
    emissions_per_kwh = df['emissions_per_kwh'].values
    battery_efficiency = np.repeat(bat_efficiency, timesteps)

    # variables
    grid_from = [pl.LpVariable(f"grid_from_{t}", 0, grid_max_rate) for t in range(timesteps)]
    grid_to = [pl.LpVariable(f"grid_to_{t}", 0, grid_max_rate) for t in range(timesteps)]
    grid_consume = [pl.LpVariable(f"grid_consume_{t}", 0, 1, pl.LpBinary) for t in range(timesteps)]

    battery_from = [pl.LpVariable(f"battery_from_{t}", 0, bat_max_rate) for t in range(timesteps)]
    battery_to = [pl.LpVariable(f"battery_to_{t}", 0, bat_max_rate) for t in range(timesteps)]
    battery_consume = [pl.LpVariable(f"battery_consume_{t}", 0, 1, pl.LpBinary) for t in range(timesteps)]

    # constraints
    M_grid = 10*grid_max_rate
    M_battery = 10*bat_max_rate

    # pareto constraint
    prob += pl.lpSum(pl.lpDot(grid_from, epex_spot)) - pl.lpSum(pl.lpDot(grid_to, epex_spot)) <= price_pareto + price_interval_pareto
    prob += pl.lpSum(pl.lpDot(grid_from, epex_spot)) - pl.lpSum(pl.lpDot(grid_to, epex_spot)) >= price_pareto - price_interval_pareto

    prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to, battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from, battery_efficiency)) == bat_tmax_soc, f'battery_soc_t0 is battery_soc_max'

    for t in range(timesteps):
        prob += pv_gen[t] + grid_from[t] + battery_from[t] == load_total[t] + grid_to[t] + battery_to[t], f'balance_{t}'

        prob += grid_from[t] - grid_consume[t]*M_grid <= 0, f'grid_from_directional_{t}'
        prob += grid_to[t] + grid_consume[t]*M_grid <= M_grid, f'grid_to_directional_{t}'
        prob += battery_from[t] - battery_consume[t] * M_battery <= 0, f'battery_from_directional_{t}'
        prob += battery_to[t] + battery_consume[t] * M_battery <= M_battery, f'battery_to_directional_{t}'

        prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to[:t+1], battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from[:t+1], battery_efficiency)) <= bat_cap, f'battery_max_soc_{t}'
        prob += bat_t0_soc + pl.lpSum(pl.lpDot(battery_to[:t+1], battery_efficiency)) - pl.lpSum(pl.lpDot(battery_from[:t+1], battery_efficiency)) >= 0, f'battery_min_soc_{t}'

    # objective
    cost = pl.lpSum(pl.lpDot(grid_from, emissions_per_kwh))
    revenue = pl.lpSum(pl.lpDot(grid_to, emissions_per_kwh))
    prob += cost - revenue

    prob.solve()
    total_cost = pl.value(prob.objective)

    results = pd.DataFrame(dict(
        grid_from=[pl.value(grid_from[t]) for t in range(timesteps)],
        grid_to=[pl.value(grid_to[t]) for t in range(timesteps)],
        grid_consume=[pl.value(grid_consume[t]) for t in range(timesteps)],
        battery_from=[pl.value(battery_from[t]) for t in range(timesteps)],
        battery_to=[pl.value(battery_to[t]) for t in range(timesteps)],
        battery_consume=[pl.value(battery_consume[t]) for t in range(timesteps)]
    ), index=df.index)

    results['soc'] = bat_t0_soc + results['battery_to'].sub(results['battery_from']).cumsum()
    results['battery'] = results['battery_from'].sub(results['battery_to'])
    results['grid'] = results['grid_from'].sub(results['grid_to'])

    return total_cost, results


bat_cap_linspace = np.linspace(0, 10000, 20)
bat_cap_costs = []
for bat_cap in bat_cap_linspace:
    cost, results = hbm_lp(bat_cap=bat_cap)
    bat_cap_costs.append(cost)

bat_cap_range = pd.Series(bat_cap_costs, index=bat_cap_linspace, name='cost').rename_axis('battery_capacity_in_kWh')
plt.figure()
bat_cap_range.plot()

max_bat_cap = 3000  # based on fig
bat_max_rate = max_bat_cap / 4

total_cost_min, results_min = hbm_lp(bat_cap=max_bat_cap, bat_max_rate=bat_max_rate)
total_cost_max, results_max = hbm_lp(bat_cap=0, bat_max_rate=bat_max_rate)
for total_cost, results in zip([total_cost_min, total_cost_max], [results_min, results_max]):
    df.join(results[['soc', 'battery', 'grid']]).plot(subplots=True, title=f'total_cost={round(total_cost,2)}')

pareto_total_cost_linespace = np.linspace(total_cost_min, total_cost_max, 50)
pareto_co2_emissions = []
for price_pareto in pareto_total_cost_linespace:
    cost, results = hbm_lp_price_pareto(bat_cap=max_bat_cap, bat_max_rate=bat_max_rate, price_pareto=price_pareto)
    pareto_co2_emissions.append(cost)

pareto_co2_emissions_range = pd.Series(pareto_co2_emissions, index=pareto_total_cost_linespace, name='co2_emissions').rename_axis('total_cost')
plt.figure()
pareto_co2_emissions_range.plot()
