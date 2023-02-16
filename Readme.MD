# Linear Optimization for profit maximization

My ongoing experimentations with linear optimization for profit maximization using battery systems integrated in a home energy management system. 

## how to run:
- Have anaconda
- open anaconda prompt, navigate to the project directory and run:
  - conda env create -f environment.yml
- run the script top to bottom in a python interpreter using the battery_linear_optimization anaconda environment

## Home energy management system:
The assumption is that there are solar panels connected to the home as well as a grid connection. 
Where I live the regulatory bodies are moving towards phasing out the possibility to feed power back into the grid by running the meter backwards.
After the phase out is complete the home-owner will receive a different (much lower) rate for feeding their generated power back into the grid. 
The scripts in this repo are my experiments with finding the optimal schedule for controlling the battery in such a manner that it can be easily adapted to the inclusion of additional constraints.
## Scripts:
- pulp_battery.py
  - This script is the basic implementation of a linear optimization scheme for the purpose described above. Synthetic data is generated to do a simulation of a single day. The PuLP library is used to carry out the simulation.
- pulp_battery_pareto_simple.py
  - My first attempt at difining a pareto frontier between cost minimization and carbon emission minimization. Effectively computing a multi-objective cost minimum given a an __alpha__ equivalency ratio between the two different objectives.
- pulp_battery_pareto.py
  - My second attempt at implementing a pareto frontier, this time aimed at making the code easier to adapt to other objectives or constraints.

## How to adapt:
- Start with understanding the pulp_battery.py script
- Adjust the values in the simulated data
- Adjust the values in the parameters
  - bat_max_rate
  - bat_cap
  - bat_t0_soc
  - bat_t_max_soc
  - bat_efficiency 
  - grid_max_rate
- Add new constraints starting on line 56

